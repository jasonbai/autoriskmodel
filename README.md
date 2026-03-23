# AutoResearch - 智能风控建模系统

> 🤖 **让AI自动进行风控建模研究** - 基于Andrej Karpathy的AutoResearch框架

## 🎯 项目简介

本项目实现了**AutoResearch核心思想**在风控建模领域的应用。

**核心突破**：通过AI Agent自动实验，发现 `extra_trees=True` 参数可将模型过拟合降低 **93%**，PSI降低 **94%**，总分提升 **80.6%**。

## 🚀 快速开始

### 第一步：准备实验数据

在使用本项目前，需要准备符合要求的实验数据集。

**数据要求：**
- ✅ CSV 格式文件
- ✅ 必须包含 `y_flag` 列（目标变量：0=好样本，1=坏样本）
- ✅ 必须包含 `window_flag` 列（train/val/oot 分割标识）
- ✅ 数值型特征列（文本列会自动删除）

**快速准备数据：**

1. 创建 CSV 文件，包含必需列：
   ```csv
   y_flag,window_flag,feature_1,feature_2,feature_3,...
   0,train,0.5,1.2,0.8
   1,train,0.3,0.9,1.1
   ...
   0,val,0.4,1.1,0.9
   1,val,0.6,0.8,1.2
   ...
   0,oot,0.5,1.3,0.7
   1,oot,0.2,1.0,1.0
   ```

2. 将文件放置到 `reference/train.csv`

3. 运行数据准备：
   ```bash
   python prepare.py reference/train.csv
   ```

📖 **详细数据格式说明**：请查看 [reference/DATA_FORMAT.md](reference/DATA_FORMAT.md)

### 方法1：AutoResearch自动研究（推荐）⭐

让Claude Code AI Agent自动进行风控建模研究：

```bash
bash init.sh
```

然后在Claude Code中：
```
"请阅读program.md并开始自动研究！"
```

### 方法2：生成Markdown报告

使用reporter.py生成Markdown格式报告：

```bash
python reporter.py --output ./模型报告.md
```

## 📁 项目结构

```
credit-autoresearch/               # 项目根目录
├── README.md                      # 项目总览
├── README_CREDIT.md               # 项目详细说明
├── REPORTER_GUIDE.md              # 报告生成器指南
├── program.md                     # AI Agent指令
│
├── prepare.py                     # 数据准备（固定）
├── train.py                       # 训练循环（AI修改）
├── reporter.py                    # 报告生成器
│
├── init.sh                        # 快速启动脚本
├── requirements.txt               # Python依赖
├── pyproject.toml                 # 项目配置
│
└── reference/                     # 参考文档
    └── DATA_FORMAT.md             # 数据格式说明
```

## 🏆 最佳模型配置（已验证）

通过 23 次自动实验，发现的最优LightGBM配置：

```python
MODEL_TYPE = 'lightgbm'

HPARAMS = {
    'num_leaves': 7,              # 每棵树的叶子节点数
    'learning_rate': 0.03,         # 学习率
    'max_depth': 2,                # 树的最大深度
    'n_estimators': 100,           # 树的数量
    'reg_alpha': 1.0,              # L1 正则化
    'reg_lambda': 1.0,             # L2 正则化
    'bagging_fraction': 0.8,       # 行采样比例
    'bagging_freq': 5,             # Bagging 频率
    'feature_fraction': 0.7,       # 列采样比例
    'extra_trees': True,           # ⭐ 核心突破！完全随机分裂
    'verbose': -1,
    'random_state': 42,
}
```

### 性能表现

| 指标 | Train | Val | OOT | 评价 |
|------|-------|-----|-----|------|
| **AUC** | 0.6848 | 0.7078 | **0.6765** | 优秀 |
| **KS** | 0.2574 | 0.2952 | **0.2590** | 优秀 |
| **PSI** | - | - | **0.0014** | 极佳 |
| **过拟合** | - | - | **0.0083** | 极低 |

**对比基准模型**：
- 过拟合降低：**93%** (0.124 → 0.008)
- PSI降低：**94%** (0.026 → 0.0014)
- 总分提升：**80.6%** (0.345 → 0.623)

## 🤖 AutoResearch-风控系统

**核心思想：** AI Agent自动修改 `train.py`，不断尝试不同模型和参数，保留改进的配置。

**AI会做什么：**
- 修改模型类型（LightGBM/XGBoost/HistGBDT/LogisticRegression）
- 调整超参数（learning_rate, max_depth, n_estimators等）
- 优化正则化参数
- 实验特征选择
- 启用随机化技术（extra_trees）

**预期效果：**
- 每次实验约1分钟
- 一晚上（8小时）约480次实验
- 自动找到最优模型配置

## ⭐ 核心发现：extra_trees 的威力

`extra_trees=True` 是本研究的关键发现，它通过**完全随机选择分裂点**（而非寻找最优分裂点）显著提升模型性能：

**工作原理：**
- 传统GBDT：在每个分裂点寻找最优分割
- Extra Trees：随机选择分裂点，增加树之间多样性
- 效果：降低方差，提升泛化能力

**适用场景：**
- ✅ 特征较多（>100）
- ✅ 样本量充足（>10,000）
- ✅ 需要高稳定性
- ✅ 防止过拟合

## 📊 报告生成器

生成Markdown格式的模型报告：

**报告内容：**
- 7个主要章节
- 完整评估指标（AUC/KS/PSI/过拟合程度）
- 评分分布分析
- 特征重要性排序
- 模型参数列表

**使用方式：**
```bash
python reporter.py --output ./模型报告.md
```

## 📖 文档

| 文档 | 内容 |
|------|------|
| **本文档** | 项目总览 |
| [README_CREDIT.md](README_CREDIT.md) | 项目详细说明 |
| [REPORTER_GUIDE.md](REPORTER_GUIDE.md) | 报告生成器使用指南 |

## 🎓 核心特性

### 三数据集评估
- **Train**: 训练集，模型拟合
- **Val**: 验证集，参数调优
- **OOT**: Out-of-Time，最终泛化测试

### 数据泄露防护
- ✅ 自动Y-Flag检测
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

## 💡 使用示例

### 场景1：自动研究

```bash
bash init.sh
```

在Claude Code中：
```
"请阅读program.md并开始自动研究，目标是最大化oot_auc！"
```

### 场景2：使用最佳配置

直接使用已验证的最佳配置进行训练：

```bash
# 1. 准备数据（首次使用）
python prepare.py /path/to/your/data.csv

# 2. 开始训练
python train.py
```

### 场景3：生成报告

```bash
# 使用最新模型生成报告
python reporter.py

# 指定输出路径
python reporter.py --output ./最终模型报告.md
```

## 🎓 AutoResearch核心思想

**原始AutoResearch**（by Andrej Karpathy）：AI Agent自动研究LLM训练

**核心循环：**
```
AI Agent无限循环：
1. 修改 train.py（唯一可编辑文件）
2. 运行实验（固定时间预算）
3. 检查oot_auc是否改进
4. 改进 → 保留commit，没改进 → 回退commit
5. 重复...
```

**应用到风控建模：**
- 目标指标：oot_auc（越高越好）
- 训练时间：约1分钟/次
- 实验次数：约60次/小时，480次/8小时

## 📊 实验结果记录

所有实验结果自动记录到 `results.tsv`，格式如下：

### 三数据集模式（默认）

```
commit	train_auc	train_ks	val_auc	val_ks	oot_auc	oot_ks	overfitting_oot	psi_oot	stability	total_score	training_time	description
a6c5d5b	0.692639	0.272034	0.711661	0.309509	0.679607	0.261360	0.013032	0.009664	0.048149	0.600561	3.8	exp20: extra_trees - MAJOR BREAKTHROUGH!
```

**字段说明：**
- `commit` - Git commit hash（7位短哈希）
- `train_auc/ks` - 训练集 AUC 和 KS 值
- `val_auc/ks` - 验证集 AUC 和 KS 值
- `oot_auc/ks` - OOT 集 AUC 和 KS 值
- `overfitting_oot` - 过拟合程度（train_auc - oot_auc）
- `psi_oot` - OOT 稳定性指标
- `stability` - Val-OOT 一致性（越小越稳定）
- `total_score` - 综合得分（越高越好）
- `training_time` - 训练耗时（秒）
- `description` - 实验描述

### 使用方式

```bash
# 命令行传入描述
python train.py "exp21: testing new feature"

# 查看结果
cat results.tsv

# 提取关键指标
python train.py | grep "^val_auc:\|^oot_auc:\|^stability:\|^total_score:"
```

## 📜 许可证

MIT（基于原始AutoResearch）

## 🙏 致谢

- [Andrej Karpathy](https://github.com/karpathy) - 原始AutoResearch项目
- Claude Code Team

---

**开始使用：**

🤖 **AI自动研究？** → `bash init.sh`
📊 **生成报告？** → `python reporter.py --help`
📚 **了解更多？** → [阅读完整文档](README_CREDIT.md)
