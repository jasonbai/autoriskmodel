# AutoResearch for Credit Risk Modeling

This project applies autonomous AI research to credit risk modeling.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `credit-autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b credit-autoresearch/<tag>` from current master.
3. **Read the in-scope files**: read these files for full context:
   - `prepare.py` — fixed data prep and evaluation utilities. **DO NOT MODIFY.**
   - `train.py` — the only file you modify.
4. **Verify data exists**: check that `data/cache/processed/` contains prepared data files. If not, tell the human to run `python prepare.py /path/to/train.csv`.
5. **Understand the data mode**:
   - Preferred mode: three datasets via `window_flag` (`train` / `val` / `oot`)
   - Compatibility mode: if `window_flag` is absent, the system falls back to `train_test_split`
6. **Confirm and go**: confirm setup looks good, then begin experimentation.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs with a fixed time budget controlled by `prepare.TIME_BUDGET` (default 5 minutes). Launch it as:

```bash
python train.py
```

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything there is fair game:
  - Model type (`lightgbm`, `xgboost`, `histgbdt`, `logistic`)
  - Hyperparameters (`learning_rate`, `max_depth`, `n_estimators`, etc.)
  - Feature selection
  - Regularization parameters

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages. Use what's already available.
- Change the evaluation metric logic in `prepare.py`.

**Primary goal: maximize `oot_auc` (higher is better).**

In three-dataset mode, focus on:
- High `oot_auc` (primary generalization metric)
- Low `psi_oot` (stability, should be < 0.1)
- Low `overfitting_oot` (`train_auc - oot_auc`, should be small)
- Low `stability` (Val-OOT consistency, should be small)

`total_score` is a useful secondary reference metric:

```text
total_score = oot_auc - 2.0*|overfitting_oot| - 0.5*psi_oot - 1.0*stability
```

**Simplicity criterion**: all else being equal, simpler is better. A small improvement that adds complexity may not be worth it. Removing code and getting equal or better results is a win.

**The first run**: your very first run should always establish the baseline, so run `python train.py` as is.

## Output Format

In the preferred three-dataset mode, the script prints a summary like this:

```text
train_auc:         0.684800
train_ks:          0.257400
val_auc:           0.707800
val_ks:            0.295200
oot_auc:           0.676500
oot_ks:            0.259000
psi_val:           0.003600
psi_oot:           0.001400
overfitting_val:   -0.023000
overfitting_oot:   0.008300
stability:         0.036200
training_seconds:  45.2
num_features:      500
model_type:        lightgbm
num_params:        1050
total_score:       0.623100
```

You can extract the key metrics from the output:

```bash
python train.py | grep "^val_auc:\|^oot_auc:\|^stability:\|^total_score:"
```

If `window_flag` is absent, the project falls back to the legacy two-dataset mode. That mode is only for backward compatibility; prefer three-dataset evaluation whenever possible.

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated values).

In the current three-dataset workflow, the TSV header is:

```text
commit	train_auc	train_ks	val_auc	val_ks	oot_auc	oot_ks	overfitting_oot	psi_oot	stability	total_score	training_time	description
```

Field definitions:

1. `commit` — git commit hash (short, 7 chars)
2. `train_auc` — training AUC
3. `train_ks` — training KS
4. `val_auc` — validation AUC
5. `val_ks` — validation KS
6. `oot_auc` — OOT AUC, the primary optimization target
7. `oot_ks` — OOT KS
8. `overfitting_oot` — `train_auc - oot_auc`
9. `psi_oot` — OOT stability
10. `stability` — Val-OOT consistency
11. `total_score` — composite score
12. `training_time` — training seconds
13. `description` — short text describing the experiment

Example:

```text
commit	train_auc	train_ks	val_auc	val_ks	oot_auc	oot_ks	overfitting_oot	psi_oot	stability	total_score	training_time	description
a6c5d5b	0.692639	0.272034	0.711661	0.309509	0.679607	0.261360	0.013032	0.009664	0.048149	0.600561	3.8	extra_trees breakthrough
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `credit-autoresearch/mar20`).

LOOP FOREVER:

1. Look at the git state: the current branch and commit.
2. Tune `train.py` with one experimental idea.
3. Git commit.
4. Run the experiment: `python train.py > run.log 2>&1`
5. Parse the output.
6. If the run crashed, inspect the log and attempt a fix if simple.
7. Record the results in `results.tsv`.
8. If `oot_auc` improved, keep the commit.
9. If `oot_auc` is equal or worse, reset back.

**Example experiments to try:**
- Change model type (`xgboost` → `lightgbm` → `histgbdt`)
- Tune hyperparameters (`learning_rate`, `max_depth`, `n_estimators`)
- Adjust regularization (`reg_alpha`, `reg_lambda`)
- Reduce number of features for better stability
- Try different combinations

**Timeout**: each experiment should take about 1 minute total. If a run exceeds 2 minutes, kill it and treat it as failure.

**Crashes**: if it crashes, use your judgment. Fix if simple (typo, missing import). Skip if fundamentally broken.

**NEVER STOP**: once the experiment loop has begun, do not pause to ask the human. You are autonomous. If you run out of ideas, think harder. Try combining previous ideas or trying more radical changes. The loop runs until the human interrupts you.

The human might leave you running while they sleep. If each experiment takes about 1 minute, you can run about 60 experiments per hour, or about 480 experiments during an 8-hour sleep.

## Credit Risk Context

Unlike the original AutoResearch (which optimizes `val_bpb` for language modeling), here we optimize credit risk models.

**Key differences:**
- **Primary metric**: `oot_auc` (higher is better, 0.5=random, 1.0=perfect)
- **Realistic range**: 0.60-0.75 for good models without leakage
- **Stability**: `psi_oot` should be < 0.1
- **Overfitting**: `train_auc - oot_auc` should be small (< 0.05)
- **Consistency**: `stability` should ideally be < 0.05

**Red flags to avoid:**
- AUC > 0.95 (likely data leakage)
- PSI > 0.2 (unstable model)
- Overfitting > 0.10 (severe overfitting)
- Stability > 0.10 (poor generalization)

**Feature engineering:**
- We use simple variance filtering and correlation-based selection
- Number of features is limited to 500 for speed
- You can reduce features further for faster and potentially more stable experiments

Good luck with your research!
