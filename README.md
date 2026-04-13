# LLM Operator Screenshot Tool

使用 `TorchDispatchMode` 在 **meta 张量**上拦截 causal LM 前向传播的完整 aten 算子序列，并将结果写入格式化的 Excel 工作簿。全程无需下载模型权重。

---

## 核心原理

```
HF config.json (几 KB)
       ↓  AutoConfig.from_pretrained
   config 对象
       ↓  AutoModelForCausalLM.from_config  +  torch.device("meta")
  模型结构（无权重，全为 meta 张量）
       ↓  TorchDispatchMode  +  ModuleTracker hooks
  aten 算子序列（形状 / dtype / 模块路径）
       ↓  openpyxl
  model_ops.xlsx
```

**meta 张量**：只携带形状和 dtype，不分配实际内存，不执行数值计算。整个追踪过程内存占用极低，即使是 671B 参数的 DeepSeek-V3 也能在几秒内完成。

---

## 安装

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
torch>=2.0.0
transformers>=4.36.0,<5.0.0
openpyxl>=3.1.0
```

> **Python 版本**：已在 Python 3.14 + torch 2.11.0 + transformers 4.57.6 上验证。

---

## 快速开始

### 命令行

```bash
# HF Hub 模型（只下载 config.json，不下载权重）
python screenshot_ops.py deepseek-ai/DeepSeek-V3-0324 --layers 4
python screenshot_ops.py deepseek-ai/DeepSeek-V3      --layers 4
python screenshot_ops.py Qwen/Qwen3-8B                --layers 4
python screenshot_ops.py Qwen/Qwen3-30B-A3B           --layers 4   # MoE 变体
python screenshot_ops.py mistralai/Mistral-7B-v0.1    --layers 2

# 本地目录（包含 config.json 即可）
python screenshot_ops.py ./hf_models/deepseek_v3      --layers 4
python screenshot_ops.py ./hf_models/llama3_8b        --layers 2
```

输出文件默认保存在当前目录，命名规则为 `<model_slug>_ops.xlsx`。

### Python API

```python
from screenshot_ops import run_trace

# 返回 (输出路径, 算子记录列表)
output_path, records = run_trace(
    model_id="deepseek-ai/DeepSeek-V3-0324",
    num_layers=4,
    batch_size=1,
    seq_len=128,
    output_path="my_output.xlsx",   # 可选，默认自动命名
)

print(f"捕获算子数: {len(records)}")
print(f"输出文件: {output_path}")

# records 是 list[dict]，每条包含:
# idx, aten_op, module_path, layer, component,
# input_shapes, input_dtypes, output_shapes, output_dtypes
```

---

## 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `model_id` | — | 必填 | HF Hub 模型 ID 或本地路径 |
| `--layers` | `-l` | `4` | 追踪的 Transformer Block 数量（2–4 即可覆盖所有算子模式） |
| `--output` | `-o` | 自动命名 | 输出 `.xlsx` 路径 |
| `--batch-size` | `-b` | `1` | dummy 输入的 batch size |
| `--seq-len` | `-s` | `128` | dummy 输入的序列长度 |

### 关于 `--layers` 的选择

| 场景 | 推荐值 |
|------|--------|
| 纯密集模型（Llama / Qwen2 / Mistral） | 2 |
| DeepSeek-V3（前 3 层为密集层，第 4 层起为 MoE） | 4 |
| Qwen3-MoE / Mixtral（第 1 层即为 MoE） | 2 |

---

## 支持的模型

### 开箱即用（在 transformers 注册表中）

| 架构 | 示例模型 ID | 注意 |
|------|------------|------|
| LLaMA / LLaMA-2 / LLaMA-3 | `meta-llama/Llama-3.1-8B` | 需 HF 授权 |
| Qwen2 / Qwen2.5 | `Qwen/Qwen2.5-7B-Instruct` | — |
| Qwen3 (dense) | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B` | — |
| Qwen3 (MoE) | `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-235B-A22B` | — |
| Mistral | `mistralai/Mistral-7B-v0.1` | — |
| Mixtral (MoE) | `mistralai/Mixtral-8x7B-v0.1` | — |
| Gemma / Gemma2 | `google/gemma-2-9b` | — |
| Phi-3 / Phi-4 | `microsoft/Phi-4` | — |

### 需要 `trust_remote_code`（自动处理）

| 架构 | 示例模型 ID | 特殊处理 |
|------|------------|---------|
| DeepSeek-V3 | `deepseek-ai/DeepSeek-V3` | MoE meta patch |
| DeepSeek-V3-0324 (V3.2) | `deepseek-ai/DeepSeek-V3-0324` | MoE meta patch |

### 本地目录支持

本地目录只需包含 `config.json`：

- **标准模型类型**（llama / qwen2 / mistral / mixtral 等）：直接支持，无需额外文件
- **自定义架构**（如 deepseek_v3）：需要在 `config.json` 中有 `auto_map` 字段，并附带 `configuration_*.py` 和 `modeling_*.py`

```
hf_models/
├── deepseek_v3/          ← config.json 含 auto_map，附带建模代码
│   ├── config.json
│   ├── configuration_deepseek.py
│   └── modeling_deepseek.py
├── llama3_8b/            ← 只需 config.json（llama 已在 transformers 注册）
│   └── config.json
└── qwen2_7b/
    └── config.json
```

---

## 输出 Excel 说明

工作簿包含 4 个 Sheet：

### Sheet 1：Model Config

模型关键配置一览，包含（有则展示，无则跳过）：

| 字段 | 说明 |
|------|------|
| `model_id` | 模型来源 |
| `model_type` | 架构类型 |
| `hidden_size` | 隐藏层维度 |
| `num_hidden_layers` | 总层数 / 追踪层数 |
| `num_attention_heads` | 注意力头数 |
| `vocab_size` | 词表大小 |
| `n_routed_experts` | MoE 路由专家数（DeepSeek） |
| `num_local_experts` | MoE 本地专家数（Mixtral） |
| `q_lora_rank` / `kv_lora_rank` | MLA 低秩维度（DeepSeek） |
| … | 其余架构特有字段按需展示 |

### Sheet 2：Operator Sequence（主表）

每行为一条 aten 算子记录，按执行顺序排列：

| 列 | 说明 |
|----|------|
| Index | 全局执行序号（从 0 起） |
| Aten Op | aten 算子全名，如 `aten.mm.default` |
| Module Path | 触发该算子的模块路径，如 `model.layers.0.self_attn.q_proj` |
| Layer | Transformer Block 编号（空表示 embedding / norm / lm_head） |
| Component | 语义组件标签（见下方组件标签说明） |
| Input Shapes | 所有输入张量的形状 |
| Input Dtypes | 输入张量的数据类型 |
| Output Shapes | 输出张量的形状 |
| Output Dtypes | 输出张量的数据类型 |

表格带自动筛选和冻结首行，支持按 Component / Layer 等维度快速过滤。

### Sheet 3：By Component

按组件标签汇总算子数量。

### Sheet 4：By Layer

按 Transformer Block 编号汇总算子数量及涉及组件。

---

## 组件标签体系

`Component` 列的标签由模块路径中的命名模式推断，与具体模型实现无关：

| 标签前缀 | 含义 | 颜色 |
|---------|------|------|
| `norm.pre_attn` | Attention 前的 LayerNorm / RMSNorm | 绿 |
| `norm.post_attn` | Attention 后的 LayerNorm / RMSNorm | 绿 |
| `norm.final` | 最终 Norm 层 | 绿 |
| `attn.q_proj` / `attn.k_proj` / … | 标准 QKV 及输出投影 | 蓝 |
| `attn.q_a_proj` / `attn.kv_a_proj` / … | MLA 低秩投影（DeepSeek） | 蓝 |
| `attn.qk_score` | QK 内积计算 | 蓝 |
| `attn.softmax` | Softmax | 蓝 |
| `attn.rope` | RoPE 位置编码 | 蓝 |
| `moe.router.*` | MoE 路由 / 门控 | 橙 |
| `moe.shared.*` | MoE 共享专家（DeepSeek） | 黄 |
| `moe.expert.*` | MoE 路由专家 MLP | 粉 |
| `ffn.gate_proj` / `ffn.up_proj` / `ffn.down_proj` | 密集 FFN 投影 | 紫 |
| `ffn.act` | 激活函数（SiLU / GeLU） | 紫 |
| `embedding` | Token Embedding | 灰 |
| `lm_head` | 语言模型输出头 | 灰 |

---

## 实现细节

### MoE Meta Patch

MoE 模块的 `forward` 通常在路由阶段调用 `.cpu().numpy()`（如 DeepSeek 的 `moe_infer`），在 meta 张量上会失败。工具会自动检测并替换为简化版 forward：

- **检测条件**：模块有 `experts: nn.ModuleList` 属性
- **简化策略**：执行 gate（捕获路由算子）→ 运行 `experts[0]`（捕获专家 MLP 算子）→ 处理 shared expert（如有）
- **返回类型适配**：通过检查原始 `forward` 源码，自动判断是返回单 tensor（DeepSeek / Qwen-MoE 风格）还是 `(hidden, router_logits)` 二元组（Mixtral 风格）

### Autocast Meta 兼容

transformers 4.50+ 的 RoPE 实现会将 tensor 的 device type 直接传给 `torch.autocast`，meta device 在 torch 2.x 中不受支持。工具在 `_apply_compat_patches()` 中将未知 device type 重定向到 `cpu`（对 meta 张量是 no-op）。

### 过滤掉的算子

以下零开销算子默认跳过，不出现在输出中：

`view`, `reshape`, `expand`, `permute`, `contiguous`, `slice`, `unsqueeze`, `squeeze`, `split`, `clone`, `arange`, `zeros`, `ones`, `full`, `tril`, `triu`, `sort`, `scatter_`, `index_put_`, …

---

## 运行测试

```bash
# 安装测试依赖
pip install pytest

# 仅本地测试（不需要网络，秒级完成）
pytest test_screenshot_ops.py -v -m "not network"

# 包含 HF Hub 测试（只下载 config，不下载权重）
pytest test_screenshot_ops.py -v

# 运行指定模型
pytest test_screenshot_ops.py -v -k "deepseek_v3"
pytest test_screenshot_ops.py -v -k "qwen3"
```

### 测试覆盖

| 测试组 | 内容 |
|--------|------|
| `TestExtractLayerIdx` | 层编号提取逻辑 |
| `TestClassifyComponent` | 组件分类（norm / attn / MLA / MoE / FFN / embedding / lm_head） |
| `TestMoEDetection` | MoE 模块检测与 patch 替换 |
| `test_local_model[*]` | 7 个本地模型端到端追踪（无网络） |
| `test_moe_components_present` | MoE 算子出现验证（DeepSeek-V3、Mixtral） |
| `test_deepseek_v3_mla_components` | MLA 专属算子出现验证 |
| `test_layer_attribution` | Block 内算子均有正确层号 |
| `test_config_summary_fields` | Config 摘要必填字段完整性 |
| `test_hub_model[*]` | HF Hub 端到端：DeepSeek-V3.2 / Qwen3-8B / Qwen3-0.6B / Qwen3-MoE |
| `test_hub_moe_detection` | Hub 模型 MoE/非MoE 检测准确性 |

---

## 项目结构

```
modeling/
├── screenshot_ops.py          # 主程序（可直接运行，也可作为库导入）
├── test_screenshot_ops.py     # pytest 自验用例
├── requirements.txt           # 依赖声明
├── deepseek_v3_ops.xlsx       # 示例输出
└── hf_models/
    ├── deepseek_v3/           # 本地模型：config + 建模代码（含 auto_map）
    ├── llama3_8b/             # 本地模型：仅 config.json
    ├── llama3_70b/
    ├── qwen2_7b/
    ├── qwen2_72b/
    ├── mistral_7b/
    ├── mixtral_8x7b/
    └── modeling_sources/      # 参考用建模源码
```
