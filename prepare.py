"""
AutoResearch for Credit Risk Modeling - Data Preparation
=========================================================
Fixed constants, one-time data prep, and runtime utilities.

This file is READ-ONLY for the AI agent. It contains:
- Training configuration (time budget, metrics)
- Data loading and preprocessing
- Feature engineering (selector.py integration)
- Model evaluation (AUC, KS, PSI)
- Y-Flag validation

Usage:
    python prepare.py  # Run once to prepare data
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# ============================================================================
# FIXED CONSTANTS - Do not modify
# ============================================================================

# Training time budget (wall clock seconds)
TIME_BUDGET = 300  # 5 minutes

# Data paths (project local)
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DATA_DIR = CACHE_DIR / "data"
PROCESSED_DIR = CACHE_DIR / "processed"

# Target variable
TARGET_COL = "y_flag"

# Feature filtering (to prevent data leakage)
DROP_PATTERNS = ['mob', 'fpd', 'dpd']  # Remove future information
DROP_COLS = ['appl_seq', 'apply_dt', 'rptno', 'id_unqf', 'id_unqp',
             'time_rep', 'timel_fsti', 'timec_fsti', 'time_fst_total', 'birth_dt']

# Feature engineering thresholds
IV_THRESHOLD = 0.02
MISSING_THRESHOLD = 0.95
IDENTICAL_THRESHOLD = 0.95
MAX_FEATURES = 500  # Limit features for faster training

# Window flag column for train/val/oot split
WINDOW_FLAG_COL = "window_flag"

# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(data_path: str = None):
    """
    One-time data preparation.

    """
    print("=" * 60)
    print("AutoResearch Credit - Data Preparation")
    print("=" * 60)

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    if data_path is None:
        # Default path (flat structure)
        data_path = Path(__file__).parent / "reference" / "train.csv"

    print(f"\nLoading data: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Original shape: {df.shape}")

    # Check target variable
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    target_dist = df[TARGET_COL].value_counts(normalize=True)
    print(f"\nTarget distribution:")
    for val, ratio in target_dist.items():
        print(f"  {val}: {ratio:.2%}")

    # Data cleaning
    print("\nData cleaning...")
    df_clean = df.copy()

    # Drop specified columns
    existing_drop_cols = [col for col in DROP_COLS if col in df_clean.columns]
    if existing_drop_cols:
        df_clean = df_clean.drop(columns=existing_drop_cols)
        print(f"  Dropped {len(existing_drop_cols)} identifier columns")

    # Drop patterns to prevent data leakage
    pattern_cols = []
    for pattern in DROP_PATTERNS:
        matched = [col for col in df_clean.columns if pattern in col.lower()
                  and col != TARGET_COL]
        if matched:
            pattern_cols.extend(matched)
            print(f"  Dropped {len(matched)} columns with '{pattern}' pattern")

    if pattern_cols:
        df_clean = df_clean.drop(columns=pattern_cols)

    # Drop object type columns (but save window_flag first)
    object_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

    # Save window_flag before dropping object columns
    window_flags = None
    if WINDOW_FLAG_COL in object_cols:
        window_flags = df_clean[WINDOW_FLAG_COL].copy()
        object_cols.remove(WINDOW_FLAG_COL)
        has_window_flag = True
    else:
        has_window_flag = WINDOW_FLAG_COL in df_clean.columns
        if has_window_flag:
            window_flags = df_clean[WINDOW_FLAG_COL].copy()

    if object_cols:
        df_clean = df_clean.drop(columns=object_cols)
        print(f"  Dropped {len(object_cols)} object columns")

    # Handle infinity
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    print(f"  Cleaned shape: {df_clean.shape}")

    # Feature selection using IV (simplified)
    print("\nFeature selection...")

    if has_window_flag:
        # Drop window_flag column for feature selection if still present
        cols_to_drop = [TARGET_COL]
        if WINDOW_FLAG_COL in df_clean.columns:
            cols_to_drop.append(WINDOW_FLAG_COL)
        X = df_clean.drop(columns=cols_to_drop)
    else:
        X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    # Simple variance-based filtering (fast)
    variances = X.var()
    high_var_cols = variances[variances > 0.001].index.tolist()
    X = X[high_var_cols]
    print(f"  Variance filtering: {len(high_var_cols)} features")

    # Limit features for speed
    if len(X.columns) > MAX_FEATURES:
        # Use simple correlation with target
        correlations = X.corrwith(y).abs()
        top_features = correlations.nlargest(MAX_FEATURES).index.tolist()
        X = X[top_features]
        print(f"  Limited to {MAX_FEATURES} features")

    if has_window_flag:
        # Use window_flag for train/val/oot split
        print("\nUsing window_flag for train/val/oot split...")

        # Split by window_flag
        train_mask = window_flags == 'train'
        val_mask = window_flags == 'val'
        oot_mask = window_flags == 'oot'

        X_train = X[train_mask]
        X_val = X[val_mask]
        X_oot = X[oot_mask]
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_oot = y[oot_mask]

        print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
        print(f"  Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
        print(f"  OOT:   {X_oot.shape[0]} samples ({X_oot.shape[0]/len(X):.1%})")
        print(f"  Features: {X_train.shape[1]}")

        # Save processed data
        print("\nSaving processed data...")
        pickle.dump(X_train, open(PROCESSED_DIR / "X_train.pkl", "wb"))
        pickle.dump(X_val, open(PROCESSED_DIR / "X_val.pkl", "wb"))
        pickle.dump(X_oot, open(PROCESSED_DIR / "X_oot.pkl", "wb"))
        pickle.dump(y_train, open(PROCESSED_DIR / "y_train.pkl", "wb"))
        pickle.dump(y_val, open(PROCESSED_DIR / "y_val.pkl", "wb"))
        pickle.dump(y_oot, open(PROCESSED_DIR / "y_oot.pkl", "wb"))
        pickle.dump(X.columns.tolist(), open(PROCESSED_DIR / "feature_names.pkl", "wb"))
        pickle.dump(True, open(PROCESSED_DIR / "has_window_flag.pkl", "wb"))
    else:
        # Fallback to train_test_split
        print("\nTrain/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        print(f"  Features: {X_train.shape[1]}")

        # Save processed data
        print("\nSaving processed data...")
        pickle.dump(X_train, open(PROCESSED_DIR / "X_train.pkl", "wb"))
        pickle.dump(X_test, open(PROCESSED_DIR / "X_test.pkl", "wb"))
        pickle.dump(y_train, open(PROCESSED_DIR / "y_train.pkl", "wb"))
        pickle.dump(y_test, open(PROCESSED_DIR / "y_test.pkl", "wb"))
        pickle.dump(X.columns.tolist(), open(PROCESSED_DIR / "feature_names.pkl", "wb"))
        pickle.dump(False, open(PROCESSED_DIR / "has_window_flag.pkl", "wb"))

    print(f"\n✅ Data preparation complete!")
    print(f"   Data saved to: {PROCESSED_DIR}")

# ============================================================================
# Runtime Utilities
# ============================================================================

def load_data():
    """
    Load preprocessed data.

    Returns:
        If has_window_flag:
            (X_train, X_val, X_oot, y_train, y_val, y_oot, feature_names, True)
        Else (backward compatible):
            (X_train, X_test, None, y_train, y_test, None, feature_names, False)
    """
    X_train = pickle.load(open(PROCESSED_DIR / "X_train.pkl", "rb"))
    y_train = pickle.load(open(PROCESSED_DIR / "y_train.pkl", "rb"))
    feature_names = pickle.load(open(PROCESSED_DIR / "feature_names.pkl", "rb"))

    has_window_flag_path = PROCESSED_DIR / "has_window_flag.pkl"
    has_window_flag = pickle.load(open(has_window_flag_path, "rb")) if has_window_flag_path.exists() else False

    if has_window_flag:
        X_val = pickle.load(open(PROCESSED_DIR / "X_val.pkl", "rb"))
        X_oot = pickle.load(open(PROCESSED_DIR / "X_oot.pkl", "rb"))
        y_val = pickle.load(open(PROCESSED_DIR / "y_val.pkl", "rb"))
        y_oot = pickle.load(open(PROCESSED_DIR / "y_oot.pkl", "rb"))
        return X_train, X_val, X_oot, y_train, y_val, y_oot, feature_names, True
    else:
        # Backward compatibility
        X_test = pickle.load(open(PROCESSED_DIR / "X_test.pkl", "rb"))
        y_test = pickle.load(open(PROCESSED_DIR / "y_test.pkl", "rb"))
        return X_train, X_test, None, y_train, y_test, None, feature_names, False

def calculate_ks(y_true, y_scores):
    """Calculate KS statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ks_value = np.max(tpr - fpr)
    return ks_value

def calculate_psi(expected, actual, buckets=10):
    """Calculate PSI (Population Stability Index)."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_percents = np.histogram(expected * 100, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual * 100, bins=breakpoints)[0] / len(actual)

    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi = np.sum((expected_percents - actual_percents) *
                 np.log(expected_percents / actual_percents))
    return psi

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance.

    Returns dict with metrics:
    - auc: Area under ROC curve
    - ks: KS statistic
    - psi: Population Stability Index
    - overfitting: train_auc - test_auc
    """
    # Predictions
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    # Metrics
    train_auc = roc_auc_score(y_train, train_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    train_ks = calculate_ks(y_train, train_pred)
    test_ks = calculate_ks(y_test, test_pred)
    psi = calculate_psi(test_pred, train_pred)

    metrics = {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_ks': train_ks,
        'test_ks': test_ks,
        'psi': psi,
        'overfitting': train_auc - test_auc
    }

    return metrics


def evaluate_model_three(model, X_train, y_train, X_val, y_val, X_oot=None, y_oot=None):
    """
    Evaluate model performance on train/val/oot datasets.

    Returns dict with metrics:
    - train_auc, train_ks: Training set metrics
    - val_auc, val_ks: Validation set metrics
    - oot_auc, oot_ks: OOT set metrics (if available)
    - psi_val, psi_oot: Population Stability Index
    - overfitting_val, overfitting_oot: Overfitting measures
    - stability: Consistency between val and oot (lower is better)
    """
    # Predictions
    train_pred = model.predict_proba(X_train)[:, 1]
    val_pred = model.predict_proba(X_val)[:, 1]

    # Train metrics
    train_auc = roc_auc_score(y_train, train_pred)
    train_ks = calculate_ks(y_train, train_pred)

    # Val metrics
    val_auc = roc_auc_score(y_val, val_pred)
    val_ks = calculate_ks(y_val, val_pred)
    psi_val = calculate_psi(val_pred, train_pred)
    overfitting_val = train_auc - val_auc

    metrics = {
        'train_auc': train_auc,
        'train_ks': train_ks,
        'val_auc': val_auc,
        'val_ks': val_ks,
        'psi_val': psi_val,
        'overfitting_val': overfitting_val,
    }

    # OOT metrics (if available)
    if X_oot is not None and y_oot is not None:
        oot_pred = model.predict_proba(X_oot)[:, 1]
        oot_auc = roc_auc_score(y_oot, oot_pred)
        oot_ks = calculate_ks(y_oot, oot_pred)
        psi_oot = calculate_psi(oot_pred, train_pred)
        overfitting_oot = train_auc - oot_auc

        # Stability: consistency between val and oot (lower is better)
        stability = max(abs(val_auc - oot_auc), abs(val_ks - oot_ks))

        metrics.update({
            'oot_auc': oot_auc,
            'oot_ks': oot_ks,
            'psi_oot': psi_oot,
            'overfitting_oot': overfitting_oot,
            'stability': stability,
        })

    return metrics

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    prepare_data(data_path)
