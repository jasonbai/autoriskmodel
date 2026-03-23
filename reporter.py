"""
AutoResearch - 智能风控报告生成器
生成Markdown格式的模型评估报告

Usage:
    # 使用最新生成的模型
    python reporter.py

    # 指定模型文件
    python reporter.py --model path/to/model.pkl

    # 指定输出路径
    python reporter.py --output my_report.md
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# ============================================================================
# 常量配置
# ============================================================================

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "credit-autoresearch")
MODEL_DIR = os.path.join(CACHE_DIR, "models")
PROCESSED_DIR = os.path.join(CACHE_DIR, "processed")

# ============================================================================
# 报告生成器类
# ============================================================================

class ModelReportGenerator:
    """模型报告生成器（Markdown格式）"""

    def __init__(self, model_path=None, data_prefix='credit'):
        """初始化报告生成器"""
        self.data_prefix = data_prefix
        self._load_data()

        if model_path is None:
            model_path = self._find_latest_model()

        self.model_path = model_path
        self._load_model(model_path)
        self._generate_predictions()
        self._calculate_scores()

    def _load_data(self):
        """加载预处理后的数据"""
        print("加载数据...")

        with open(os.path.join(PROCESSED_DIR, f"{self.data_prefix}_X_train.pkl"), 'rb') as f:
            self.X_train = pickle.load(f)
        with open(os.path.join(PROCESSED_DIR, f"{self.data_prefix}_X_test.pkl"), 'rb') as f:
            self.X_test = pickle.load(f)
        with open(os.path.join(PROCESSED_DIR, f"{self.data_prefix}_y_train.pkl"), 'rb') as f:
            self.y_train = pickle.load(f)
        with open(os.path.join(PROCESSED_DIR, f"{self.data_prefix}_y_test.pkl"), 'rb') as f:
            self.y_test = pickle.load(f)
        with open(os.path.join(PROCESSED_DIR, f"{self.data_prefix}_feature_names.pkl"), 'rb') as f:
            self.feature_names = pickle.load(f)

        print(f"  训练集: {self.X_train.shape}")
        print(f"  测试集: {self.X_test.shape}")
        print(f"  特征数: {len(self.feature_names)}")

    def _find_latest_model(self):
        """找到最新的模型文件"""
        model_files = list(Path(MODEL_DIR).glob("*.pkl"))
        if not model_files:
            raise FileNotFoundError(f"在 {MODEL_DIR} 中没有找到模型文件")

        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"使用最新模型: {latest_model}")
        return str(latest_model)

    def _load_model(self, model_path):
        """加载模型"""
        print(f"加载模型: {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # 从文件名提取模型类型
        self.model_name = Path(model_path).stem.split('_')[0]
        print(f"  模型类型: {self.model_name}")

    def _generate_predictions(self):
        """生成预测"""
        print("生成预测...")

        self.train_pred = self.model.predict_proba(self.X_train)[:, 1]
        self.test_pred = self.model.predict_proba(self.X_test)[:, 1]

        self.train_pred_class = self.model.predict(self.X_train)
        self.test_pred_class = self.model.predict(self.X_test)

    def _calculate_scores(self):
        """计算评分（将概率转换为分数）"""
        print("计算评分...")

        # 使用logit转换：score = -log(odds) * factor + base_score
        epsilon = 1e-6
        train_odds = self.train_pred / (1 - self.train_pred + epsilon) + epsilon
        test_odds = self.test_pred / (1 - self.test_pred + epsilon) + epsilon

        # 转换为分数（PDO方法）
        base_score = 600
        pdo = 20
        factor = pdo / np.log(2)

        self.train_score = base_score - factor * np.log(train_odds)
        self.test_score = base_score - factor * np.log(test_odds)

        # 限制分数范围到合理区间
        self.train_score = np.clip(self.train_score, 300, 900)
        self.test_score = np.clip(self.test_score, 300, 900)

    # ========================================================================
    # 评估指标计算
    # ========================================================================

    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        auc = roc_auc_score(y_true, y_pred)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ks = np.max(tpr - fpr)

        return {'auc': auc, 'ks': ks}

    def calculate_psi(self, expected, actual, buckets=10):
        """计算PSI（Population Stability Index）"""
        breakpoints = np.linspace(0, 100, buckets + 1)
        expected_percents = np.histogram(expected * 100, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual * 100, bins=breakpoints)[0] / len(actual)

        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        psi = np.sum((expected_percents - actual_percents) *
                     np.log(expected_percents / actual_percents))
        return psi

    # ========================================================================
    # 生成Markdown报告
    # ========================================================================

    def generate_report(self, output_path=None):
        """生成Markdown格式的报告"""

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"model_report_{timestamp}.md"

        print(f"\n生成Markdown报告: {output_path}")

        # 构建报告内容
        report_content = self._build_report_content()

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✅ 报告已生成: {output_path}")
        return output_path

    def _build_report_content(self):
        """构建报告内容"""
        lines = []

        # 标题
        lines.append("# 模型评估报告\n")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**模型类型**: {self.model_name}\n")
        lines.append(f"**模型路径**: {self.model_path}\n")

        # 目录
        lines.append("\n---\n")
        lines.append("## 目录\n")
        lines.append("1. [数据概览](#数据概览)")
        lines.append("2. [模型性能](#模型性能)")
        lines.append("3. [模型稳定性](#模型稳定性)")
        lines.append("4. [评分分布](#评分分布)")
        lines.append("5. [特征重要性](#特征重要性)")
        lines.append("6. [模型参数](#模型参数)")

        # 数据概览
        lines.append("\n---\n")
        lines.append("## 数据概览\n")
        lines.append("### 数据集统计\n")
        lines.append("| 数据集 | 样本数 | 坏样本数 | 坏样本率 |")
        lines.append("|--------|--------|----------|----------|")

        train_n = len(self.y_train)
        train_bad = int(self.y_train.sum())
        train_rate = self.y_train.mean()

        test_n = len(self.y_test)
        test_bad = int(self.y_test.sum())
        test_rate = self.y_test.mean()

        lines.append(f"| 训练集 | {train_n:,} | {train_bad:,} | {train_rate:.2%} |")
        lines.append(f"| 测试集 | {test_n:,} | {test_bad:,} | {test_rate:.2%} |")

        lines.append(f"\n**特征数量**: {len(self.feature_names)}\n")

        # 模型性能
        lines.append("\n---\n")
        lines.append("## 模型性能\n")
        lines.append("### 评估指标\n")
        lines.append("| 指标 | 训练集 | 测试集 |")
        lines.append("|------|--------|--------|")

        train_metrics = self.calculate_metrics(self.y_train, self.train_pred)
        test_metrics = self.calculate_metrics(self.y_test, self.test_pred)

        lines.append(f"| AUC | {train_metrics['auc']:.4f} | {test_metrics['auc']:.4f} |")
        lines.append(f"| KS | {train_metrics['ks']:.4f} | {test_metrics['ks']:.4f} |")

        overfitting = train_metrics['auc'] - test_metrics['auc']
        lines.append(f"\n**过拟合程度**: {overfitting:.4f} (train_auc - test_auc)\n")

        # 性能评价
        lines.append("### 性能评价\n")

        if test_metrics['auc'] >= 0.75:
            auc_level = "优秀"
        elif test_metrics['auc'] >= 0.65:
            auc_level = "良好"
        elif test_metrics['auc'] >= 0.60:
            auc_level = "一般"
        else:
            auc_level = "需改进"

        lines.append(f"- **AUC评级**: {auc_level} ({test_metrics['auc']:.4f})\n")

        if test_metrics['ks'] >= 0.30:
            ks_level = "优秀"
        elif test_metrics['ks'] >= 0.20:
            ks_level = "良好"
        else:
            ks_level = "较弱"

        lines.append(f"- **KS评级**: {ks_level} ({test_metrics['ks']:.4f})\n")

        # 模型稳定性
        lines.append("\n---\n")
        lines.append("## 模型稳定性\n")
        lines.append("### PSI分析\n")

        psi = self.calculate_psi(self.train_score, self.test_score)

        if psi < 0.1:
            psi_level = "优秀 ✅"
        elif psi < 0.2:
            psi_level = "良好 ⚠️"
        else:
            psi_level = "需要关注 ❌"

        lines.append(f"- **总体PSI**: {psi:.4f} - {psi_level}\n")

        lines.append("\n**PSI评判标准**:\n")
        lines.append("- PSI < 0.1: 稳定性变化很小 ✅")
        lines.append("- 0.1 ≤ PSI < 0.2: 有变化 ⚠️")
        lines.append("- PSI ≥ 0.2: 变化显著 ❌\n")

        # 评分分布
        lines.append("\n---\n")
        lines.append("## 评分分布\n")
        lines.append("### 等频分箱统计\n")

        # 使用训练集确定分位数
        breakpoints = np.percentile(self.train_score, np.linspace(0, 100, 11))

        lines.append("| 分箱 | 评分区间 | 样本数 | 样本占比 | 坏样本数 | 逾期率 | Lift |")
        lines.append("|------|----------|--------|----------|----------|--------|------|")

        for i in range(len(breakpoints) - 1):
            lower = breakpoints[i]
            upper = breakpoints[i + 1]

            # 测试集中落在该区间的样本
            mask = (self.test_score >= lower) & (self.test_score < upper)
            n_samples = mask.sum()
            n_bad = (self.y_test[mask]).sum()
            bad_rate = n_bad / n_samples if n_samples > 0 else 0
            lift = bad_rate / self.y_test.mean() if self.y_test.mean() > 0 else 0

            lines.append(f"| {i+1} | [{lower:.1f}, {upper:.1f}) | {n_samples} | {n_samples/test_n:.2%} | {int(n_bad)} | {bad_rate:.2%} | {lift:.2f} |")

        lines.append(f"\n**评分范围**: {self.test_score.min():.1f} - {self.test_score.max():.1f}\n")

        # 特征重要性
        lines.append("\n---\n")
        lines.append("## 特征重要性\n")
        lines.append("### Top 20 特征\n")

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            lines.append("\n该模型不支持特征重要性提取\n")
            importance = None

        if importance is not None:
            # 排序
            indices = np.argsort(importance)[::-1]
            total_importance = importance.sum()
            top_n = min(20, len(importance))

            lines.append("| 排名 | 特征名称 | 重要性 | 占比 |")
            lines.append("|------|----------|--------|------|")

            for i in range(top_n):
                idx = indices[i]
                imp = importance[idx]
                pct = imp / total_importance

                # 限制特征名长度
                feat_name = self.feature_names[idx]
                if len(feat_name) > 30:
                    feat_name = feat_name[:27] + "..."

                lines.append(f"| {i+1} | {feat_name} | {imp:.4f} | {pct:.2%} |")

        # 模型参数
        lines.append("\n---\n")
        lines.append("## 模型参数\n")

        if hasattr(self.model, 'get_params'):
            params = self.model.get_params()

            # 只显示主要参数
            key_params = []
            if self.model_name == 'LightGBM':
                key_params = ['num_leaves', 'learning_rate', 'max_depth', 'n_estimators', 'reg_alpha', 'reg_lambda']
            elif self.model_name == 'XGBoost':
                key_params = ['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight', 'subsample', 'colsample_bytree']
            elif self.model_name == 'HistGBDT':
                key_params = ['max_depth', 'learning_rate', 'max_iter', 'min_samples_leaf']
            elif self.model_name == 'LogisticRegression':
                key_params = ['C', 'penalty', 'max_iter']

            lines.append("| 参数 | 值 |")
            lines.append("|------|-----|")

            for key in key_params:
                if key in params:
                    value = params[key]
                    lines.append(f"| {key} | {value} |")

        # 总结
        lines.append("\n---\n")
        lines.append("## 总结\n")

        lines.append("### 模型优势\n")
        if test_metrics['auc'] >= 0.70:
            lines.append("- ✅ 排序能力强（AUC ≥ 0.70）")
        else:
            lines.append("- ⚠️ 排序能力中等（AUC < 0.70）")

        if psi < 0.1:
            lines.append("- ✅ 模型稳定性优秀（PSI < 0.1）")
        else:
            lines.append("- ⚠️ 模型稳定性需关注（PSI ≥ 0.1）")

        if overfitting < 0.05:
            lines.append("- ✅ 过拟合程度低（< 0.05）")
        else:
            lines.append("- ⚠️ 存在一定过拟合（≥ 0.05）")

        lines.append("\n### 建议\n")

        if test_metrics['auc'] < 0.65:
            lines.append("- 考虑增加特征工程或调整模型参数")

        if overfitting > 0.10:
            lines.append("- 建议增加正则化强度或减少模型复杂度")

        if psi > 0.1:
            lines.append("- 建议重新训练或调整特征选择策略")

        lines.append("\n---\n")
        lines.append(f"\n*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("\n*AutoResearch v1.0 - 智能风控系统*")

        return "\n".join(lines)

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AutoResearch智能风控 - 报告生成器（Markdown格式）"
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='模型文件路径（默认使用最新模型）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出Markdown文件路径（默认：model_report_YYYYMMDD_HHMMSS.md）'
    )
    parser.add_argument(
        '--data-prefix',
        type=str,
        default='credit',
        help='数据文件前缀'
    )

    args = parser.parse_args()

    print("="*60)
    print("AutoResearch 智能风控 - Markdown报告生成器")
    print("="*60)

    # 创建报告生成器
    generator = ModelReportGenerator(
        model_path=args.model,
        data_prefix=args.data_prefix
    )

    # 生成报告
    output_path = generator.generate_report(args.output)

    print("\n" + "="*60)
    print("✅ 报告生成完成！")
    print("="*60)
    print(f"\n文件位置: {output_path}")

if __name__ == "__main__":
    main()
