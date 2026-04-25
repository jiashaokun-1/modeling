# Session Progress

## 当前阶段：modeller step_time 修复 — 已完成

## 最新变更（2026-04-25）

- 已检查提交 `a4f8c6c6cb5b3ed6f4e23ec6bc0a8794fd11ecd7` 中的 `python/zrt/transform/analysis/modeller.py` 实现。
- 已恢复图原生训练建模入口：`TrainingReport`、`estimate_training()`、`estimate_training_from_graphs()`、`model_training()`。
- 已恢复 `python/zrt/transform/analysis/__init__.py` 中的 modeller API 导出。
- 已在 `python/zrt/ir/adapter.py::stitch_fwd_bwd()` 元数据中补充 `fwd_bwd_stitched=True`，解除 `TrainingMemoryPass` graph-native activation 分支的 metadata gate。
- 已撤回被中断轮次中对 `python/zrt/training/compose/stage.py` 的无关临时改动。
- 已修复 `python/zrt/cli.py::_run_training_modelling()` 绕过 modeller 的问题：`--train --hw` 现在调用 `estimate_training_from_graphs()`，由 modeller 执行 graph-native stitch + training pipeline。
- `estimate_training_from_graphs()` / `model_training()` 新增 `cp` 透传，避免 CLI `--cp` 在恢复后的 graph-native 路径中丢失。
- 新增 `tests/training/test_cli_modeller_wiring.py`，防止 CLI 退回旧的 forward/backward 分离 pipeline。
- 已修复 `estimate_training_from_graphs()` 重算 `step_time_ms` 的结构性错误：现在直接读取 `TrainingPipelinePass` 的 `pipeline_metrics.step_time_ms` / `mfu` / bubble 与 warmup/cooldown/steady 指标。
- `estimate_training_from_graphs()` / `model_training()` 新增 `pp_schedule`、`vpp_chunks` 参数，graph-native modeller 入口可选择 VPP/DualPipe 调度。
- `tests/training/test_captured_graph_modelling.py` 新增 DualPipe schedule-adjusted step_time 回归测试，防止退回 `(M + pp - 1) * per_stage_ms` 简化公式。

## 本轮验证

```
python -m py_compile python/zrt/transform/analysis/modeller.py python/zrt/transform/analysis/__init__.py python/zrt/training/compose/stage.py
PYTHONPATH=python python -c "from python.zrt.transform.analysis import TrainingReport, estimate_training, estimate_training_from_graphs, model_training; print('analysis exports ok')"
PYTHONPATH=python pytest tests/training/test_captured_graph_modelling.py -q
python -m py_compile python/zrt/cli.py python/zrt/transform/analysis/modeller.py tests/training/test_cli_modeller_wiring.py
PYTHONPATH=python pytest tests/training/test_cli_modeller_wiring.py -q
PYTHONPATH=python pytest tests/training/test_graph_schedule.py -q
python -m py_compile python/zrt/transform/analysis/modeller.py tests/training/test_captured_graph_modelling.py
PYTHONPATH=python pytest tests/training/test_graph_schedule.py tests/training/test_cli_modeller_wiring.py -q
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -q
```

结果：`test_cli_modeller_wiring.py` 1 passed；`test_captured_graph_modelling.py` 16 passed；`test_graph_schedule.py` 5 passed。

已知剩余风险：`tests/training/anchors/test_anchors.py -q` 目前 12 passed / 1 failed，失败为 GPT-3 strict MFU calibration gap（estimated=0.2264，anchor=0.5200，deviation=56.5%）。本轮修复 graph-native modeller 的 step_time 传播问题，但 spec anchor 校准仍需后续 P3/P2 工作继续处理。

参考计划：`/Users/sky/.claude/plans/based-on-the-above-bright-hopcroft.md`
补充计划：`/Users/sky/.claude/plans/details-of-the-content-lively-stonebraker.md`

## 所有子项完成状态

| 子项 | 状态 | 说明 |
|------|------|------|
| 4.0/4.1/4.2 spec 路径 Composer | ✅ 完成 | `compose/pipeline.py` 4 个 Composer 类 |
| 4.1 VPP 测试（spec 路径） | ✅ 完成 | `test_interleaved_1f1b.py` |
| 4.2 DualPipe 测试（spec 路径） | ✅ 完成 | `test_dualpipe.py` |
| 4.3 EP 负载不均衡 | ✅ 完成 | `compose/stage.py::ep_imbalance_factor` |
| 4.4 搜索 / Pareto | ✅ 完成 | `search/space.py` + `estimator.py` |
| 4.5 Anchor 验证 | ✅ 完成 | `anchor/validate.py` + 3 个 YAML + `test_anchors.py` |
| 4.6 Chrome Trace | ✅ 完成 | `training/trace/exporter.py` + `report/chrome_trace.py` + `report/summary.py` chrome_trace 字段 + CLI `--trace` |
| **图路径调度分派** | ✅ 完成 | `training.py:285-298` + `modeller.py:307-343` + `test_graph_schedule.py` |

## 完成的变更（2026-04-24，第二轮）

### 变更 1：图路径调度分派（CRITICAL）

- **`python/zrt/transform/context.py`**：`TrainingConfig` 新增 `vpp_chunks: int = 1`
- **`python/zrt/transform/analysis/training.py:285-298`**：替换硬编码 1F1B 为按 `ctx.training.pp_schedule` 分派（interleaved / dualpipev / dualpipe / 1f1b）
- **`python/zrt/transform/analysis/modeller.py:307-343`**：unified 路径直接读取 `pipeline_metrics.step_time_ms`，不再重新计算
- **`tests/training/test_graph_schedule.py`**：5 个图路径测试（VPP/DualPipe/DualPipeV/bubble_fraction/pp1）全部通过

### 变更 2：Chrome Trace 报告集成（MEDIUM）

- **`python/zrt/report/chrome_trace.py`**：新建，`build_chrome_trace(timeline) -> dict`
- **`python/zrt/report/summary.py`**：`TrainingSummary` 新增 `chrome_trace: dict | None = None`
- **`python/zrt/training/cli.py`**：`model-training` 子命令新增 `--trace <path>` 参数

### 变更 3：Anchor YAML 文件（LOW）

- **`tests/training/anchors/gpt3_175b_megatron.yaml`**：GPT-3 175B，H100 SXM，TP8 PP1 DP64，MFU 0.52
- **`tests/training/anchors/llama3_70b_meta.yaml`**：LLaMA-3 70B，TP4 PP2 DP16，MFU 0.48
- **`tests/training/anchors/deepseek_v3.yaml`**：DeepSeek-V3，TP8 EP64 PP16，MFU 0.35
- **`tests/training/anchors/test_anchors.py`**：8 个测试（YAML 格式验证 + anchor/report 集成），全部通过

## 全量测试结果

```
252 passed, 34 warnings in 55.84s
```

零回归，无失败。

## 历史里程碑摘要

- Phase 0：`stitch_fwd_bwd()` 前向+反向图拼接（69/69 training tests pass）
- Phase 1：步骤时间公式修复 + 激活内存 + FLOPs 修复
- Phase 2：`PipelineParallelPass` + 逐阶段 `DAGScheduler` + 1F1B 公式
- Phase 3：`context_parallel.py` / `data_parallel.py` / CoC/MC2 overlap 注解
- Phase 4（完成）：spec 路径 Composer ✅；Chrome Trace ✅；图路径调度分派 ✅；EP 不均衡 ✅；搜索/Pareto ✅；Anchor 验证 ✅
