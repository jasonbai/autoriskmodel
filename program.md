# AutoResearch for Credit Risk Modeling

This is an experiment to apply autonomous AI research to credit risk modeling.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `credit-autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b credit-autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed data prep, evaluation utilities. **DO NOT MODIFY.**
   - `train.py` — the file you modify. Model architecture, hyperparameters, training loop.
4. **Verify data exists**: Check that `data/cache/processed/` contains data files. If not, tell the human to run `python prepare.py /path/to/train.csv`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs with a **fixed time budget** (default 5 minutes). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - Model type (lightgbm, xgboost, histgbdt, logistic)
  - Hyperparameters (learning_rate, max_depth, n_estimators, etc.)
  - Feature selection (use top N features)
  - Regularization parameters

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages. Use what's already available.
- Change the evaluation metric. The `evaluate_model` function is the ground truth.

**The goal: maximize test_auc (higher is better).** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Focus on:
- High test_auc (sorting capability)
- Low psi (stability, should be < 0.1)
- Low overfitting (train_auc - test_auc, should be small)

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds complexity may not be worth it. Removing code and getting equal or better results is a win.

**The first run**: Your very first run should always be to establish the baseline, so run `python train.py` as is.

## Output format

Once the script finishes it prints a summary like this:

```
test_auc:          0.640000
test_ks:           0.150000
psi:               0.000100
overfitting:       0.010000
training_seconds:  45.2
num_features:      500
model_type:        lightgbm
num_params:        3100
total_score:       0.619450
```

You can extract the key metric from the output:
```
python train.py | grep "^test_auc:\|^psi:\|^overfitting:\|^total_score:"
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated values).

The TSV has a header row and 5 columns:

```
commit	test_auc	psi	overfitting	description
```

1. git commit hash (short, 7 chars)
2. test_auc achieved (e.g. 0.650000) — use 0.000000 for crashes
3. psi (e.g. 0.000100) — use 9.999999 for crashes
4. overfitting (e.g. 0.010000) — use 9.999999 for crashes
5. short text description of what this experiment tried

Example:
```
commit	test_auc	psi	overfitting	description
a1b2c3d	0.640000	0.000100	0.010000	baseline lightgbm
b2c3d4e	0.655000	0.000050	0.008000	increase n_estimators to 200
c3d4e5f	0.638000	0.000200	0.015000	switch to xgboost
d4e5f6g	0.000000	9.999999	9.999999	crash OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `credit-autoresearch/mar20`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1`
5. Read out the results: parse the output
6. If the run crashed, check the log and attempt a fix
7. Record the results in results.tsv
8. If test_auc improved (higher), keep the commit
9. If test_auc is equal or worse, git reset back

**Example experiments to try:**
- Change model type (xgboost → lightgbm → histgbdt)
- Tune hyperparameters (learning_rate, max_depth, n_estimators)
- Adjust regularization (reg_alpha, reg_lambda)
- Reduce number of features (faster training, less overfitting)
- Try different combinations

**Timeout**: Each experiment should take ~1 minute total. If a run exceeds 2 minutes, kill it and treat as failure.

**Crashes**: If it crashes, use your judgment. Fix if simple (typo, missing import). Skip if fundamentally broken.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. You are autonomous. If you run out of ideas, think harder — try combining previous ideas, try more radical changes. The loop runs until the human interrupts you.

The human might leave you running while they sleep. If each experiment takes ~1 minute, you can run ~60 experiments per hour, for ~480 experiments during an 8-hour sleep.

## Credit Risk Context

Unlike the original AutoResearch (which optimizes val_bpb for language modeling), we're optimizing credit risk models:

**Key differences:**
- **Metric**: test_auc (higher is better, 0.5=random, 1.0=perfect)
- **Realistic range**: 0.60-0.75 for good models (without data leakage)
- **Stability**: psi should be < 0.1 (population stability)
- **Overfitting**: train_auc - test_auc should be small (< 0.05)

**Red flags to avoid:**
- AUC > 0.95 (likely data leakage)
- PSI > 0.2 (unstable model)
- Overfitting > 0.10 (severe overfitting)

**Feature engineering:**
- We use simple variance filtering and correlation-based selection
- Number of features is limited to 500 for speed
- You can reduce features further for faster experiments

Good luck with your research!
