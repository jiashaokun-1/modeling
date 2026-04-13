# LLM Operator Tracer

通过 `TorchDispatchMode` 在 **meta 张量**上追踪 HuggingFace LLM 模型的 `aten` 算子调用序列，输出格式化的 Excel 报告和算子融合规则 JSON。无需 GPU。

## 功能

- 拦截模型 forward 过程中触发的每一个 `aten` 算子
- 自动过滤零成本的 reshape/view 类算子
- 记录每个算子的模块路径、层级、组件分类、输入输出 shape/dtype
- 两阶段自动算子融合（leaf modules → parent modules）
- 数据流分析：区分融合 kernel 的外部 I/O 与内部张量
- 输出 6 个 Sheet 的 Excel 报告 + 融合规则 JSON
- 支持 DeepSeek-V3 和 DeepSeek-V3.2

## 快速开始

```bash
pip install torch transformers openpyxl

# 追踪 V3
python screenshot_ops.py
python -m screenshot_ops.main --model v3

# 追踪 V3.2
python -m screenshot_ops.main --model v3.2

# 自定义参数
python -m screenshot_ops.main --model v3.2 --layers 8 --seq-len 256
```

运行后生成：

| 文件 | 内容 |
|---|---|
| `deepseek_v3_ops.xlsx` | V3 可视化报告 |
| `deepseek_v3_2_ops.xlsx` | V3.2 可视化报告 |
| `*_fusion_rules.json` | 自动发现的算子融合模式 |

## Excel 报告结构

| Sheet | 说明 |
|---|---|
| Model Config | 模型配置摘要 |
| Fused Operators | 融合后的算子序列（主视图，含融合 I/O 映射） |
| Raw Operator Sequence | 原始 aten 算子完整序列 |
| Summary | 按融合算子聚合统计 |
| By Layer | 按层级聚合统计 |
| Fusion Rules | 自动发现的融合模式（含融合 I/O 映射） |

## 工作原理

1. 从 `hf_models/{deepseek_v3,deepseek_v3_2}/config.json` 加载配置
2. 修补 `transformers` 内部工具函数以兼容旧版建模代码
3. 在 **meta 设备**上实例化模型（默认 4 层：0-2 为 dense，3+ 为 MoE）
4. 替换 `MoE.forward` 和 `Indexer.forward` 以绕过 meta 张量上报错
5. 在 `TorchDispatchMode` 下执行一次 forward，拦截所有 aten 算子
6. 过滤 `SKIP_OPS` 中定义的零成本算子
7. 两阶段融合：先按 leaf module 分组，再向上合并到 parent module（单 parent 最多 30 个子算子）
8. 数据流分析：通过张量 ID 追踪，区分融合组的外部 I/O 与内部传递张量
9. 写入 Excel + JSON

## V3 vs V3.2 对比

| 特性 | V3 | V3.2 |
|---|---|---|
| 原始算子数 (4层) | 400 | 468 |
| 融合后算子数 | 75 | 87 |
| 融合模式数 | 6 | 7 |
| Indexer 模块 | 无 | 有 (MLA 注意力中新增) |

## 项目结构

```
├── screenshot_ops.py                    # 入口（转发到 package）
├── screenshot_ops/
│   ├── __init__.py
│   ├── main.py                          # 入口：加载模型、追踪、输出
│   ├── dispatch.py                      # RecordingDispatch + TensorTracker
│   ├── tracker.py                       # ModuleTracker（forward hooks）
│   ├── fusion.py                        # FusionEngine + FusionSpec
│   ├── model_loader.py                  # 模型加载 + 兼容性修补
│   ├── classifier.py                    # 组件分类 + 颜色映射
│   ├── excel_writer.py                  # Excel + JSON 输出
│   └── tensor_utils.py                  # 张量工具 + SKIP_OPS
├── hf_models/
│   ├── deepseek_v3/                     # DeepSeek-V3 本地代码与配置
│   ├── deepseek_v3_2/                   # DeepSeek-V3.2 本地代码与配置
│   ├── modeling_sources/                # 其他架构的参考建模文件
│   ├── llama3_70b/
│   ├── llama3_8b/
│   ├── mistral_7b/
│   ├── mixtral_8x7b/
│   ├── qwen2_72b/
│   └── qwen2_7b/
├── deepseek_v3_ops.xlsx                 # V3 生成产物
├── deepseek_v3_2_ops.xlsx               # V3.2 生成产物
├── deepseek_v3_ops_fusion_rules.json
├── deepseek_v3_2_ops_fusion_rules.json
└── README.md
```

## 融合 I/O 映射

每个融合条目都标注了：

- **Fused Input Shapes/Dtypes**：融合 kernel 的外部输入（被组内消费但非组内产生的张量）
- **Input Sources**：每个输入来自哪个子算子的哪个输入端口
- **Fused Output Shapes/Dtypes**：融合 kernel 的外部输出（在组内产生但未被组内消费的张量）
- **Output Sources**：每个输出由哪个子算子的哪个输出端口产生

## 注意事项

- **仅 meta 设备**：不加载真实权重，捕获的是算子结构而非运行时行为
- **Monkey-patching**：脚本在导入时修补 `transformers` 内部函数和模型 forward 方法
- **`__init__.py` 临时文件**：导入期间在 `hf_models/` 下临时创建再删除，使模型目录被识别为 package
- **`rope_scaling` 键**：`PretrainedConfig.__init__` 可能改写该字段，脚本会提前保存并恢复

## 依赖

- `torch`
- `transformers`
- `openpyxl`
