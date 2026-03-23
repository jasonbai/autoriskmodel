# AutoResearch - 智能风控建模系统

> 🤖 让 AI 自动进行风控建模研究，基于 Andrej Karpathy 的 AutoResearch 思路。

## 项目简介

本项目把 AutoResearch 的“自动试验、自动比较、持续优化”流程应用到信用风险建模场景中。当前主流程基于三数据集评估：

- `train`：训练集
- `val`：验证集
- `oot`：时间外测试集

AI Agent 的主要工作是围绕 `train.py` 持续做实验，目标是提升 `oot_auc`，同时兼顾稳定性、过拟合和 Val-OOT 一致性。

## 快速开始

### 1. 准备数据

数据至少需要包含：

- `y_flag`：目标变量
- 数值型特征列
- 推荐提供 `window_flag`：启用 `train/val/oot` 三数据集评估

运行数据准备：

```bash
python prepare.py reference/train.csv
```

更完整的数据格式说明见 `reference/DATA_FORMAT.md`。

### 2. 运行自动研究

```bash
bash init.sh
```

然后在 Claude Code 中执行：

```text
请阅读 program.md 并开始自动研究，目标是最大化 oot_auc！
```

### 3. 生成模型报告

```bash
python reporter.py --output ./模型报告.md
```

说明：当 `--output` 传相对路径时，报告会写入 `report/` 目录。

## 核心文件

```text
prepare.py      # 数据准备与评估工具，固定不改
train.py        # 训练与实验主入口，AI 可修改
reporter.py     # Markdown 报告生成器
program.md      # AutoResearch 实验指令
```

## 文档导航

- `README_CREDIT.md`：项目详细说明，包括历史最佳实验配置、评估指标、结果记录格式和完整工作流
- `REPORTER_GUIDE.md`：报告生成器使用指南
- `reference/DATA_FORMAT.md`：输入数据格式说明
- `program.md`：AI 自动实验指令

## 当前工作流

1. 用 `prepare.py` 处理原始数据
2. 用 `train.py` 训练并输出 `train/val/oot` 指标
3. 自动将实验结果记录到 `results.tsv`
4. 用 `reporter.py` 生成 Markdown 报告

## 许可证

Apache-2.0，详见 `LICENSE`。

## 致谢

- [Andrej Karpathy](https://github.com/karpathy)
- Claude Code Team
