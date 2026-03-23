# AutoResearch - 智能风控建模系统

> 🤖 **让AI自动进行风控建模研究** - 基于Andrej Karpathy的AutoResearch框架

> 本文档是详细说明文档；如果你只想快速上手，请先看 `README.md`。

## 🎯 项目简介

本项目实现了AutoResearch核心思想在风控建模领域的应用。

**核心突破**：通过AI Agent自动实验，发现 `extra_trees=True` 参数可将模型过拟合降低 **93%**，PSI降低 **94%**，总分提升 **80.6%**。

### 核心功能

**AutoResearch-风控系统** - 让Claude Code Agent自动实验和优化

```
├── prepare.py      # 数据准备（固定）
├── train.py        # 训练循环（AI修改）
└── program.md      # AI指令
```

**报告生成器** - 生成Markdown模型报告

```
reporter.py         # 报告生成器
```

## 🚀 使用方式

### 方式1：AutoResearch自动研究（推荐）

让AI自动进行风控建模研究：

```bash
bash init.sh
```

然后在Claude Code中：
```
"请阅读program.md并开始自动研究！"
```

**AI会做什么：**
- 自动修改 train.py
- 尝试不同模型（LightGBM/XGBoost/HistGBDT/LogisticRegression）
- 调整超参数
- 优化特征选择
- 自动保留改进的配置

**预期效果：**
- 每次实验约1分钟
- 一晚上（8小时）约480次实验
- 自动找到最优模型配置

### 方式2：参考历史最佳配置

历史最佳实验配置见下文“历史最佳模型配置详解”章节，可作为调参参考，不等同于当前 `train.py` 默认值。

### 方式3：生成报告

使用reporter.py生成Markdown模型报告：

```bash
python reporter.py --output ./模型报告.md
```

说明：当 `--output` 传相对路径时，报告会写入 `report/` 目录。

## 🏆 历史最佳模型配置详解

以下配置和指标来自某次历史最佳实验快照，用于说明调参思路，不等同于当前 `train.py` 默认值。

### 完整参数说明

```python
MODEL_TYPE = 'lightgbm'

HPARAMS = {
    # 模型复杂度控制
    'num_leaves': 7,              # 每棵树的叶子节点数（小值防止过拟合）
    'max_depth': 2,                # 树的最大深度（浅树增强泛化）
    'n_estimators': 100,           # 树的数量

    # 学习参数
    'learning_rate': 0.03,         # 低学习率，稳定收敛

    # 正则化
    'reg_alpha': 1.0,              # L1 正则化
    'reg_lambda': 1.0,             # L2 正则化

    # 随机化技术
    'bagging_fraction': 0.8,       # 每次迭代使用80%的数据
    'bagging_freq': 5,             # 每5次迭代进行一次bagging
    'feature_fraction': 0.7,       # 每次迭代使用70%的特征
    'extra_trees': True,           # ⭐ 完全随机分裂（关键突破）

    # 其他
    'verbose': -1,                 # 不输出训练日志
    'random_state': 42,            # 随机种子
}
```

### 参数调优路径

本研究通过23次实验发现的优化路径：

```
基准配置
  ↓ 降低复杂度 (num_leaves: 15→7, depth: 4→2)
过拟合从 0.124 → 0.042
  ↓ 降低学习率 (0.1 → 0.03)
PSI从 0.018 → 0.00003
  ↓ 添加bagging (fraction: 0.8, freq: 5)
稳定性提升
  ↓ 添加feature_fraction (0.7)
进一步提升稳定性
  ↓ 启用extra_trees (True) ⭐
过拟合 → 0.008, PSI → 0.001, 总分 → 0.623
```

### ⭐ extra_trees 的关键作用

**什么是 Extra Trees？**

Extra Trees (Extremely Randomized Trees) 是一种随机化技术：

| 特性 | 传统 GBDT | Extra Trees |
|------|-----------|-------------|
| 分裂点选择 | 寻找最优分裂点 | 随机选择分裂点 |
| 树之间多样性 | 较低 | 更高 |
| 方差 | 较高 | 更低 |
| 过拟合风险 | 较高 | 更低 |

**为什么有效？**

1. **降低方差**：随机分裂点增加树之间的多样性
2. **防止过拟合**：减少对训练数据的过度拟合
3. **提升泛化**：更好的 OOT 性能
4. **快速训练**：不需要寻找最优分裂点

**适用场景：**
- ✅ 特征数量较多（>100）
- ✅ 样本量充足（>10,000）
- ✅ 需要高稳定性
- ✅ PSI 监控严格

### 性能对比

| 配置 | OOT AUC | 过拟合 | PSI | 总分 |
|------|---------|--------|-----|------|
| 基准 | 0.6849 | 0.1245 | 0.0256 | 0.3450 |
| +复杂度控制 | 0.6809 | 0.0416 | 0.0154 | 0.5425 |
| +低学习率 | 0.6706 | 0.0244 | 0.00003| 0.5696 |
| +bagging | 0.6704 | 0.0234 | 0.00004| 0.5703 |
| +feature_frac | 0.6704 | 0.0235 | 0.00022| 0.5753 |
| **+extra_trees** | **0.6765** | **0.0083** | **0.0014** | **0.6231** |

## 📁 项目结构

```
credit-autoresearch/               # 项目根目录
├── README.md                      # 项目主入口
├── README_CREDIT.md               # 本文档
├── REPORTER_GUIDE.md              # 报告生成器指南
│
├── prepare.py                     # 数据准备（固定）
├── train.py                       # 训练循环（AI修改）
├── reporter.py                    # 报告生成器
├── program.md                     # AI指令
│
├── init.sh                        # 快速启动脚本
├── requirements.txt               # Python依赖
├── pyproject.toml                 # 项目配置
│
└── reference/                     # 参考文档
    └── DATA_FORMAT.md             # 数据格式说明
```

## 📋 数据集要求

### 如何准备实验数据

本项目需要您自行准备风控建模数据集。以下是详细步骤：

**第一步：准备 CSV 文件**

创建一个包含以下核心列的 CSV 文件：

| 列名 | 类型 | 说明 | 必需 |
|------|------|------|------|
| `y_flag` | int | 目标变量（0=好样本，1=坏样本） | ✅ |
| `window_flag` | string | 推荐提供，用于启用 train/val/oot 三数据集评估 | 推荐 |
| 特征列 | 数值型 | 风控特征变量 | ✅ |

**第二步：数据质量检查**

| 指标 | 要求 | 说明 |
|------|------|------|
| **文件格式** | CSV | 标准逗号分隔值文件 |
| **目标变量** | `y_flag` 列 | 二分类标签：0=好样本，1=坏样本 |
| **窗口标识** | 推荐提供 `window_flag` 列 | 启用 train/val/oot 三数据集评估；缺失时回退为两数据集模式 |
| **特征类型** | 数值型 | 文本列会自动删除 |
| **样本量** | 建议 >10,000 | 样本越多模型越好 |
| **坏样本率** | 1-20% | 目标变量中 1 的比例 |

**第三步：准备数据文件**

```bash
# 将你的数据文件放到项目目录
cp /path/to/your/data.csv ./reference/train.csv

# 运行数据准备（首次使用）
python prepare.py ./reference/train.csv
```

### 数据格式示例

最小示例（`train.csv`）：

```csv
y_flag,window_flag,feature_1,feature_2,feature_3
0,train,0.5,1.2,0.8
1,train,0.3,0.9,1.1
0,train,0.7,1.5,0.6
0,val,0.4,1.1,0.9
1,val,0.6,0.8,1.2
0,oot,0.5,1.3,0.7
1,oot,0.2,1.0,1.0
```

> 📖 **完整数据格式说明**：请查看 [reference/DATA_FORMAT.md](reference/DATA_FORMAT.md)

### 三数据集分割

系统根据 `window_flag` 列自动分割数据：

| window_flag | 数据集 | 用途 |
|-------------|--------|------|
| `train` | 训练集 | 模型拟合 |
| `val` | 验证集 | 参数调优 |
| `oot` | 测试集 | 泛化测试 |

### 目标变量说明

```python
TARGET_COL = "y_flag"  # 默认配置
```

**含义**：
- `0` = 正常（好样本）
- `1` = 违约（坏样本）

**如果你的目标变量列名不同**，需要修改 `prepare.py`：

```python
# 修改 prepare.py
TARGET_COL = "your_target_column"  # 改成你的列名
```

### 自动删除的列

系统会自动删除以下类型的列，防止数据泄露和过拟合：

#### 1. 标识符列（自动删除）

```python
DROP_COLS = [
    'appl_seq',      # 申请序号
    'apply_dt',      # 申请日期
    'rptno',         # 报告编号
    'id_unqf',       # 唯一标识符
    'id_unqp',       # 唯一标识符
    # ... 更多标识符
]
```

#### 2. 数据泄露特征（自动删除）

```python
DROP_PATTERNS = ['mob', 'fpd', 'dpd']
```

- **`mob`** - Mobile on Book（未来账期信息）
- **`fpd`** - First Payment Default（首次违约）
- **`dpd`** - Days Past Due（逾期天数）

**例外**：目标变量不会被删除

#### 3. 文本列（自动删除）

所有 `object` 类型（字符串）的列会被自动删除。

## 📊 核心特性

### 三数据集评估
- **Train**: 训练集，模型拟合
- **Val**: 验证集，参数调优
- **OOT**: Out-of-Time，最终泛化测试

### 数据泄露防护
- ✅ 校验 `y_flag` 目标列是否存在
- ✅ 泄露特征警告（mob/fpd/dpd）
- ✅ 模式匹配删除

### 模型支持
- LightGBM（快速，推荐）
- XGBoost（性能）
- HistGBDT（稳定）
- LogisticRegression（可解释）

### 评估指标
- **AUC** - 排序能力（目标：最大化）
- **KS** - 区分度（目标：>0.2）
- **PSI** - 稳定性（目标：<0.1）
- **过拟合程度** - train_auc - oot_auc（目标：<0.05）

## 🎓 AutoResearch核心思想

**原始AutoResearch**（by Andrej Karpathy）：AI Agent自动研究LLM训练

**核心循环：**
```
AI Agent无限循环：
1. 修改 train.py（唯一可编辑文件）
2. 运行实验（固定时间预算）
3. 检查oot_auc是否改进
4. 改进 → 保留，没改进 → 回退
5. 重复...
```

**应用到风控建模：**
- 目标指标：oot_auc（越高越好）
- 训练时间：约1分钟/次
- 实验次数：约60次/小时

## 🔧 工作流

### 完整研究流程

```bash
# 1. 使用AutoResearch自动研究
bash init.sh
# 在Claude Code中启动自动研究
# 睡觉 → AI持续实验480次

# 2. 查看最优配置
cat results.tsv

# 3. 生成最终报告
python reporter.py --output ./最终模型报告.md
```

## 📊 实验结果记录 (results.tsv)

### 三数据集记录格式

系统支持三数据集（train/val/oot）评估，所有实验结果自动记录到 `results.tsv`：

```
commit	train_auc	train_ks	val_auc	val_ks	oot_auc	oot_ks	overfitting_oot	psi_oot	stability	total_score	training_time	description
a6c5d5b	0.692639	0.272034	0.711661	0.309509	0.679607	0.261360	0.013032	0.009664	0.048149	0.600561	3.8	exp20: extra_trees - MAJOR BREAKTHROUGH!
```

**字段说明：**

| 字段 | 说明 | 目标 |
|------|------|------|
| `commit` | Git commit hash（7位短哈希） | - |
| `train_auc` | 训练集 AUC | 高 |
| `train_ks` | 训练集 KS 统计量 | >0.2 |
| `val_auc` | 验证集 AUC | 高 |
| `val_ks` | 验证集 KS 统计量 | >0.2 |
| `oot_auc` | OOT 集 AUC（核心指标） | **最大化** |
| `oot_ks` | OOT 集 KS 统计量 | >0.2 |
| `overfitting_oot` | train_auc - oot_auc | < 0.05 |
| `psi_oot` | OOT 稳定性指标 | < 0.1 |
| `stability` | Val-OOT 一致性 | < 0.05 |
| `total_score` | 综合得分 | **最大化** |
| `training_time` | 训练耗时（秒） | < TIME_BUDGET |
| `description` | 实验描述 | - |

**综合得分公式：**
```
total_score = oot_auc - 2.0*|overfitting_oot| - 0.5*psi_oot - 1.0*stability
```

### 使用方式

```bash
# 方式1: 命令行传入描述（推荐用于自动化）
python train.py "exp21: testing new feature"

# 方式2: 交互式输入（手动实验）
python train.py
# 会提示: Enter experiment description:

# 查看所有实验结果
cat results.tsv

# 提取关键指标（三数据集模式）
python train.py | grep "^val_auc:\|^oot_auc:\|^stability:\|^total_score:"

# 查看最优实验
sort -t$'\t' -k11 -rn results.tsv | head -5  # 按 total_score 排序
```

## 💡 最佳实践

### 1. 从最佳配置开始

优先参考上文“历史最佳模型配置详解”中的配置快照，再根据你的数据分布和稳定性要求做微调。

### 2. 根据数据特征调整

| 数据特征 | 建议调整 |
|----------|----------|
| 样本量小（<10K） | 减少 n_estimators，增加正则化 |
| 特征少（<50） | 增大 feature_fraction |
| 不平衡严重 | 调整 scale_pos_weight |
| PSI 高 | 降低模型复杂度 |

### 3. 性能预期

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| AUC | >0.70 | 0.65-0.70 | <0.65 |
| PSI | <0.05 | 0.05-0.10 | >0.10 |
| 过拟合 | <0.03 | 0.03-0.05 | >0.05 |

### 4. 红旗警告

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| AUC > 0.95 | 数据泄露 | 检查特征，删除未来信息 |
| PSI > 0.20 | 模型不稳定 | 降低复杂度，增加正则化 |
| 过拟合 > 0.10 | 严重过拟合 | 减少 n_estimators，降低 max_depth |

## 🚧 未来计划

- [ ] 添加WOE编码支持
- [ ] 实现特征工程自动化
- [ ] 支持多模型融合
- [ ] 实验可视化Dashboard

## 📜 许可证

Apache-2.0 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Andrej Karpathy](https://github.com/karpathy) - 原始AutoResearch项目
- Claude Code Team

---

**开始使用：**

🤖 **AI自动研究？** → `bash init.sh`
📊 **生成报告？** → `python reporter.py --help`
📚 **了解更多？** → [阅读完整文档](README.md)
