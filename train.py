"""
AutoResearch for Credit Risk Modeling - Training Loop
=====================================================
This is the ONLY file the AI agent modifies.

Contains:
- Model architecture (LightGBM/XGBoost/HistGBDT/LogisticRegression)
- Hyperparameters
- Training loop
- Model configuration

Everything is fair game for the AI agent to modify:
- Model type and architecture
- Hyperparameters (learning rate, depth, etc.)
- Feature engineering
- Regularization

The only constraint: training must finish within TIME_BUDGET (5 minutes).
"""

import time
import pickle
import warnings
import numpy as np
import subprocess

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

import lightgbm as lgb
import xgboost as xgb

# Import fixed utilities from prepare.py
import prepare

# ============================================================================
# MODEL CONFIGURATION - Agent modifies this section
# ============================================================================

MODEL_TYPE = 'lightgbm'  # Options: 'lightgbm', 'xgboost', 'histgbdt', 'logistic'

# Model hyperparameters
HPARAMS = {
    'num_leaves': 7,
    'learning_rate': 0.03,
    'max_depth': 2,
    'n_estimators': 150,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'feature_fraction': 0.7,
    'extra_trees': True,
    'verbose': -1,
    'random_state': 42,
}

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(X_train, y_train, X_val, y_val):
    """
    Train the model with configured hyperparameters.

    Returns the trained model.
    """
    print(f"Training {MODEL_TYPE} model...")

    if MODEL_TYPE == 'lightgbm':
        model = lgb.LGBMClassifier(**HPARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(period=False)]
        )

    elif MODEL_TYPE == 'xgboost':
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='auc',
            **HPARAMS
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    elif MODEL_TYPE == 'histgbdt':
        # Map parameters
        hist_params = {
            'max_depth': HPARAMS.get('max_depth', 5),
            'learning_rate': HPARAMS.get('learning_rate', 0.1),
            'max_iter': HPARAMS.get('n_estimators', 100),
            'random_state': 42
        }
        model = HistGradientBoostingClassifier(**hist_params)
        model.fit(X_train, y_train)

    elif MODEL_TYPE == 'logistic':
        # Map parameters
        log_params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        }
        model = LogisticRegression(**log_params)
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    return model

def log_to_results(metrics, score, training_time, description):
    """
    Log experiment results to results.tsv.

    For three-dataset mode, includes train/val/oot AUC and KS.
    """
    import os
    from pathlib import Path

    results_file = Path(__file__).parent / "results.tsv"

    # Get git commit hash
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except:
        commit = "unknown"

    # Prepare row data
    if 'oot_auc' in metrics:
        # Three-dataset mode
        row = {
            'commit': commit,
            'train_auc': f"{metrics['train_auc']:.6f}",
            'train_ks': f"{metrics['train_ks']:.6f}",
            'val_auc': f"{metrics['val_auc']:.6f}",
            'val_ks': f"{metrics['val_ks']:.6f}",
            'oot_auc': f"{metrics['oot_auc']:.6f}",
            'oot_ks': f"{metrics['oot_ks']:.6f}",
            'overfitting_oot': f"{metrics['overfitting_oot']:.6f}",
            'psi_oot': f"{metrics['psi_oot']:.6f}",
            'stability': f"{metrics['stability']:.6f}",
            'total_score': f"{score:.6f}",
            'training_time': f"{training_time:.1f}",
            'description': description
        }
    else:
        # Two-dataset mode (backward compatible)
        row = {
            'commit': commit,
            'train_auc': f"{metrics['train_auc']:.6f}",
            'train_ks': f"{metrics['train_ks']:.6f}",
            'test_auc': f"{metrics['test_auc']:.6f}",
            'test_ks': f"{metrics['test_ks']:.6f}",
            'overfitting': f"{metrics['overfitting']:.6f}",
            'psi': f"{metrics['psi']:.6f}",
            'total_score': f"{score:.6f}",
            'training_time': f"{training_time:.1f}",
            'description': description
        }

    # Write to file
    file_exists = results_file.exists()

    with open(results_file, 'a') as f:
        if not file_exists:
            # Write header for three-dataset mode
            if 'oot_auc' in metrics:
                header = '\t'.join([
                    'commit', 'train_auc', 'train_ks', 'val_auc', 'val_ks',
                    'oot_auc', 'oot_ks', 'overfitting_oot', 'psi_oot',
                    'stability', 'total_score', 'training_time', 'description'
                ])
            else:
                header = '\t'.join([
                    'commit', 'train_auc', 'train_ks', 'test_auc', 'test_ks',
                    'overfitting', 'psi', 'total_score', 'training_time', 'description'
                ])
            f.write(header + '\n')

        # Write data row
        f.write('\t'.join(row.values()) + '\n')

    print(f"\n✅ Results logged to {results_file}")

def main():
    """Main training loop."""
    import time
    import sys

    print("=" * 60)
    print("AutoResearch Credit - Training")
    print("=" * 60)

    # Load data (new 8-value return)
    print("\nLoading data...")
    X_train, X_val, X_oot, y_train, y_val, y_oot, feature_names, has_window_flag = prepare.load_data()

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    if has_window_flag and X_oot is not None:
        print(f"  OOT:   {X_oot.shape}")

    # Handle missing values for all datasets
    print("\nHandling missing values...")
    all_X = [X_train, X_val]
    if X_oot is not None:
        all_X.append(X_oot)

    for col in X_train.columns:
        has_na = any(X[col].isna().any() for X in all_X)
        if has_na:
            col_min = min(X[col].min() for X in all_X)
            fill_val = col_min - 100
            for X in all_X:
                X[col] = X[col].fillna(fill_val)

    # Print model config
    print(f"\nModel Configuration:")
    print(f"  Type: {MODEL_TYPE}")
    print(f"  Hyperparameters: {HPARAMS}")

    # Train with time budget
    print(f"\nTraining for max {prepare.TIME_BUDGET} seconds...")
    start_time = time.time()

    model = train_model(X_train, y_train, X_val, y_val)

    training_time = time.time() - start_time
    print(f"  Training time: {training_time:.1f}s")

    # Evaluate with appropriate function
    print("\nEvaluating...")
    if has_window_flag:
        metrics = prepare.evaluate_model_three(
            model, X_train, y_train, X_val, y_val, X_oot, y_oot
        )
    else:
        metrics = prepare.evaluate_model(model, X_train, y_train, X_val, y_val)

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    if has_window_flag:
        # Three-dataset output
        print(f"train_auc:         {metrics['train_auc']:.6f}")
        print(f"train_ks:          {metrics['train_ks']:.6f}")
        print(f"val_auc:           {metrics['val_auc']:.6f}")
        print(f"val_ks:            {metrics['val_ks']:.6f}")
        print(f"oot_auc:           {metrics['oot_auc']:.6f}")
        print(f"oot_ks:            {metrics['oot_ks']:.6f}")
        print(f"psi_val:           {metrics['psi_val']:.6f}")
        print(f"psi_oot:           {metrics['psi_oot']:.6f}")
        print(f"overfitting_val:   {metrics['overfitting_val']:.6f}")
        print(f"overfitting_oot:   {metrics['overfitting_oot']:.6f}")
        print(f"stability:         {metrics['stability']:.6f}")
        print(f"training_seconds:  {training_time:.1f}")
        print(f"num_features:      {len(feature_names)}")
        print(f"model_type:        {MODEL_TYPE}")
        print(f"num_params:        {count_params(model)}")

        # New scoring formula: prioritize OOT performance with stability penalty
        score = (
            metrics['oot_auc']
            - 2.0 * abs(metrics['overfitting_oot'])
            - 0.5 * metrics['psi_oot']
            - 1.0 * metrics['stability']
        )
        print(f"total_score:       {score:.6f}")
    else:
        # Backward compatibility output
        print(f"train_auc:         {metrics['train_auc']:.6f}")
        print(f"test_auc:          {metrics['test_auc']:.6f}")
        print(f"test_ks:           {metrics['test_ks']:.6f}")
        print(f"psi:               {metrics['psi']:.6f}")
        print(f"overfitting:       {metrics['overfitting']:.6f}")
        print(f"training_seconds:  {training_time:.1f}")
        print(f"num_features:      {len(feature_names)}")
        print(f"model_type:        {MODEL_TYPE}")
        print(f"num_params:        {count_params(model)}")

        score = metrics['test_auc'] - 2.0 * abs(metrics['overfitting']) - 0.5 * metrics['psi']
        print(f"total_score:       {score:.6f}")

    # Log to results.tsv
    print("\n" + "=" * 60)

    # Get description from command line args or use default
    description = None
    if len(sys.argv) > 1:
        description = ' '.join(sys.argv[1:])

    if description is None:
        try:
            description = input("Enter experiment description: ").strip()
        except EOFError:
            description = "auto_experiment"

    if not description:
        description = "auto_experiment"

    log_to_results(metrics, score, training_time, description)

    return metrics, score

def count_params(model):
    """Count approximate number of parameters."""
    if hasattr(model, 'coef_'):
        return sum(p.size for p in [model.coef_, model.intercept_] if p is not None)
    elif hasattr(model, 'feature_importances_'):
        # For tree models, approximate based on trees * leaves
        if hasattr(model, 'n_features_in_'):
            if MODEL_TYPE == 'lightgbm' or MODEL_TYPE == 'xgboost':
                n_estimators = model.n_estimators if hasattr(model, 'n_estimators') else 100
                num_leaves = HPARAMS.get('num_leaves', 31)
                return n_estimators * num_leaves
        return model.n_features_in_
    return 0

if __name__ == "__main__":
    main()
