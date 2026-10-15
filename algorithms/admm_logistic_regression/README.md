# ADMM Logistic Regression

**Python module:** `admm_logistic_regression.py` — a single module, following this repo's
usual algorithm layout (see `logistic_regression/`, `coxph/`, `fed_statistics/`, etc.): a
`central()` orchestrator plus `@data(1)` node-side (partial) functions, no separate package or
`setup.py`.

This algorithm is designed to be run with the [vantage6](https://vantage6.ai) infrastructure for
distributed analysis and learning. See the "ADMM logistic regression" section in the repo root
[README.md](../../README.md) for how to build the image and run it against this harness.

## ⚠️ This algorithm only does federated learning, not data preparation

`central()` takes `feature_cols` and `target_col` as arguments (see `run_study.py`), exactly like
`logistic_regression/`. It assumes:

- Every column in `feature_cols` is **already numeric** on every node.
- `target_col` is **already a binary (0/1) column** on every node.
- All of these columns exist, under the **exact same names**, on every node's data.

This algorithm does **not** encode categorical variables, derive outcomes from other columns, or
validate your data beyond checking the above columns exist (missing columns raise a clear
`ValueError`; wrong *content* — e.g. a non-numeric feature, or a target that isn't actually binary
— will not raise, it will just silently produce wrong results). Preparing data to this contract is
the caller's responsibility — use `data/datavalgen` (schema validation) or your own pipeline, not
this algorithm, for that. See `data/generate_beach_data.py` for a worked example: it generates raw
BEACH-schema clinical data (text stage labels, a vital-status + follow-up-days outcome) *and*
precomputes clean numeric/binary columns (`t_stage_code`, `n_stage_code`, `m_stage_code`,
`overall_stage_code`, `two_year_survival`) from it, so this algorithm never has to see the raw
columns.

## How it differs from `logistic_regression/`

Both algorithms fit a logistic regression model federatedly on the same kind of input (numeric
`feature_cols` + binary `target_col`), but with different optimization strategies:

| | `logistic_regression/` | `admm_logistic_regression/` (this one) |
|---|---|---|
| Optimization | Federated averaging (FedAvg): each node runs local SGD, the central node averages weights across nodes each round | ADMM consensus: each node solves a *full local optimization* against a shared consensus vector `z`; the central node updates `z` (and dual variables) in closed form each round |
| Relationship to a pooled fit | An approximation — accuracy depends on rounds/local epochs/batch size and can drift from the pooled solution | Converges to (approximately, within tolerance) the **same coefficients as a centralized/pooled fit** on the combined data — the core motivation for using ADMM at all |
| Stopping | Fixed number of rounds | Fixed number of rounds *or* early stop once primal/dual residuals pass tolerance (a real convergence check, not just "ran out of rounds") |
| Regularization | None | Optional L1 shrinkage via `lambda_` (intercept excluded) |
| Normalization | Federated mean/std (Phase 1), z-score applied before training/evaluation | Same mechanism — a `compute_stats` phase computes global mean/std, applied identically in training, evaluation, and final prediction |
| Train/validation split | Random per-node sample (`train_ratio` + seed) | Same mechanism (`train_ratio` + seed) |
| Evaluation | Accuracy (correct / n), per-node and global | Accuracy, RMSE, global + per-site ROC/AUC, and a logistic recalibration (slope/intercept) plus a quantile calibration curve |

In short: use `logistic_regression/` for a general-purpose federated logistic regression on
arbitrary numeric features; use this one when you specifically need the trained model to match
what a centralized/pooled analysis would have produced, which is what ADMM consensus guarantees
(within numerical tolerance) and FedAvg does not. Both now share the same data contract, so
switching between them only means changing `ALGORITHM_IMAGE` and the `method` in `run_study.py`.

## Provenance

The ADMM consensus approach here is **adapted from**:

- **GitHub:** [mdw-nl/20kChallengeVantage6](https://github.com/mdw-nl/20kChallengeVantage6)
  (`my-fl-project/20kLogRegChallenge`), which is itself conceptually based on
- **GitHub:** [RadiationOncologyOntology/20kChallenge](https://github.com/RadiationOncologyOntology/20kChallenge/)
- **Manuscript:** *Distributed learning on 20 000+ lung cancer patients – The Personal Health
  Train* (Deist et al.; see authors on the
  [20kChallenge README](https://github.com/RadiationOncologyOntology/20kChallenge/)).

The original `RadiationOncologyOntology/20kChallenge` repository implements training/validation
of logistic regression and summary statistics in a **master/site** architecture
(MATLAB + Varian Learning Portal). `mdw-nl/20kChallengeVantage6` ported that to a Python/vantage6
setting as a `setup.py`-packaged algorithm specific to BEACH-schema clinical data
(`logreg_challenge_20k/{central,partial}.py`), which did its own TNM-stage-label mapping,
one-hot encoding, and two-year-survival outcome derivation internally.

This version keeps the ADMM consensus math (`_logistic_admm_objective`, `_z_objective`,
`_check_convergence` — unchanged, see `test_admm_logistic_regression.py`'s gradient-check tests)
but **generalizes everything around it**: BEACH-specific preprocessing was removed from the
algorithm entirely (moved to `data/generate_beach_data.py`, since it's a data-preparation concern,
not a federated-learning one) in favor of the generic `feature_cols`/`target_col` contract
described above, matching `logistic_regression/`'s interface. It also adds a federated
normalization phase (previously absent, since the original BEACH stage-bucket features were
already on a small, similar scale — arbitrary numeric features are not), drops a
`logging`-toggle kwarg in favor of this repo's usual always-on `info()` logging, and renames
`central_function` → `central` to match this repo's other algorithms.
`test_admm_approximately_matches_pooled_logistic_regression_predictions` is the test that actually
proves the core ADMM-vs-pooled-fit property survived all of this.
