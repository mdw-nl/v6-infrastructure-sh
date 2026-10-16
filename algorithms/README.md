# Algorithms

This covers the workflow shared by every algorithm in this folder — building images, running
studies, overriding variables, the full data-contracts table, and adding your own. See the root
[README.md](../README.md) for what each algorithm actually does and the infrastructure harness
itself; see each algorithm's own README (where one exists — `admm_logistic_regression/`,
`argos_cnn/`) for details specific to that one algorithm.

## Data contracts

This is deliberately documented here, not only in each algorithm's source file — if you're
running one of these from its **published image** (see **Using a published image instead of
building** below) rather than a local checkout, this table is what you actually have access to.

Every algorithm raises a clear `ValueError` (naming the missing column(s) and the dataset's
actual available columns) if a configured column doesn't exist on a node — none of them
silently skip a misconfigured column. None of them do any categorical encoding, outcome
derivation, or data cleaning themselves; preparing data to the contract below is the caller's
responsibility (see the root README's **Data** section and `data/datavalgen/` for how this
repo's own test datasets do that).

| Algorithm | Kwargs | Data contract |
|---|---|---|
| `average` | `column` | Numeric column, same name on every node. |
| `logistic_regression` | `feature_cols`, `target_col` | `feature_cols`: numeric. `target_col`: already binary (0/1). |
| `admm_logistic_regression` | `feature_cols`, `target_col` | Same contract as `logistic_regression` — see its own README for the full comparison between the two. |
| `coxph` | `time_col`, `outcome_col`, `expl_vars` | `time_col`: non-negative numeric duration. `outcome_col`: binary 0/1 event indicator (1 = event, 0 = censored). `expl_vars`: numeric covariates. Nodes with ≤10 observed events are excluded from the analysis (a per-node privacy threshold, not configurable via kwarg). |
| `kaplan_meier` | `time_col`, `event_col` | Same time/event contract as `coxph`. Optional Gaussian/Poisson noise injection on event times (`noise_type`/`snr`/`random_seed`) is a privacy mechanism, not a data requirement. |
| `fed_statistics` | `statistics` (dict of `{column: [stat, ...]}`) | Any column present under that name on every node — no fixed schema. `counts`/`binned_counts` work on any column type; `minmax`/`mean`/`std`/`quantiles` require the column to be numeric (or numeric-coercible), else a clear `TypeError`. |
| `argos_cnn` | `patient_col`/`slice_col`/`ct_col`/`gt_col`/`label_col`, `num_classes`, `num_channels`, `min_bound`/`max_bound`, `patch_size`, `seed` | CT only (HU windowing is CT-specific). Axial slices, consistently oriented and resampled to the same voxel spacing across every patient/node — **never validated** (the code doesn't read a NIfTI's affine); violating this produces silently wrong results, not an error. Every slice must be exactly `patch_size` x `patch_size` (divisible by 32) — a mismatch now raises a clear error. GT label values must be integers in `[0, num_classes)`. See `argos_cnn/README.md` and `argos_cnn/argos_cnn.py`'s module docstring for the full detail. |

## Python environment (uv)

A single `pyproject.toml` at the **repo root** covers all dependencies for both `algorithms/` (run study scripts, lint algorithm code) and `data/` (synthetic data generation, e.g. `datavalgen`). Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment from the repo root:

```bash
uv sync
```

This creates one `.venv` at the repo root, shared by both `algorithms/` and `data/`. Activate it or prefix commands with `uv run` from the root (e.g. `uv run python algorithms/average/run_study.py`).

## Building an algorithm image

Each algorithm must be built as a Docker image before the nodes can execute it. Build from the repo root so the tag matches what `run_study.py` expects:

```bash
# Average
docker build -t average:latest algorithms/average/

# Logistic regression
docker build -t logistic_regression:latest algorithms/logistic_regression/

# Kaplan-Meier
docker build -t kaplan_meier:latest algorithms/kaplan_meier/

# Cox proportional hazards
docker build -t coxph:latest algorithms/coxph/

# Federated descriptive statistics
docker build -t fed_statistics:latest algorithms/fed_statistics/
```

Because the nodes run inside Docker on the same host daemon, no registry push is needed for local testing. If nodes report `non-existing Docker image` anyway, see the root README's **Local image registry** section.

## Using a published image instead of building

You don't have to build any algorithm image yourself. Every algorithm in `algorithms/` is built and published automatically by `.github/workflows/build-algorithm-images.yaml` on every merge to `main`, as a public image at:

```text
ghcr.io/mdw-nl/v6-infrastructure-sh/<algorithm-folder-name>:latest
```

e.g. `ghcr.io/mdw-nl/v6-infrastructure-sh/average:latest`, `ghcr.io/mdw-nl/v6-infrastructure-sh/argos_cnn:latest`. These are public — no `docker login` or credentials needed to pull them.

To use one instead of building locally, just point that algorithm's `run_study.py` at the full registry reference instead of the bare local tag:

```python
# instead of:
ALGORITHM_IMAGE = "average:latest"
# use:
ALGORITHM_IMAGE = "ghcr.io/mdw-nl/v6-infrastructure-sh/average:latest"
```

No code or Dockerfile changes needed — `run_study.py` just passes this string straight through as the task's `image`, and the vantage6 **node** (not your shell) pulls it from GHCR itself the first time a task references it, the same way it already pulls the vantage6 server/node/UI images. Skip the `docker build` step for that algorithm entirely. Each of the 7 current algorithms (`average`/`logistic_regression`/`admm_logistic_regression`/`coxph`/`fed_statistics`/`kaplan_meier`/`argos_cnn`) has been verified to run correctly this way.

Also tag with a specific commit SHA (visible in the package's version list on GitHub, or as the second tag the workflow pushes alongside `:latest`) instead of `:latest` if you want to pin to an exact, reproducible build rather than whatever the most recent merge to `main` currently points at.

## Running a study

With the infrastructure up (`infra.sh up`) and the image built:

```bash
# from the repo root
uv run python algorithms/average/run_study.py
uv run python algorithms/logistic_regression/run_study.py
uv run python algorithms/kaplan_meier/run_study.py
uv run python algorithms/coxph/run_study.py
uv run python algorithms/fed_statistics/run_study.py
```

The script authenticates against the local vantage6 server, submits the central task, waits for all nodes to complete, and prints the results.

## Changing variables without rebuilding the image

To test a different column/variable (e.g. `VARIABLE` in `average/run_study.py`, `FEATURE_COLS`/`TARGET_COL` in `logistic_regression/run_study.py`, or `TIME_COL`/`OUTCOME_COL`/`EXPL_VARS` in `coxph/run_study.py`), just edit the constants at the top of that algorithm's `run_study.py` and re-run it — no image rebuild needed. The constants in the algorithm file itself (`average.py`, etc.) are only the fallback defaults.

`coxph/` runs against the same LUNG1 data as `kaplan_meier/` (`Survival.time`/`deadstatus.event`), with `age`, `clinical.T.Stage`, `Clinical.N.Stage`, `Clinical.M.Stage` as the default covariates — no new node profile needed. It ships a `test_math_correctness.py` (run with `uv run pytest algorithms/coxph/`) that validates the federated Newton-Raphson fit against a directly-fit reference Cox model on a bundled test fixture — the actual proof the ported math wasn't broken in translation, independent of any live vantage6 infrastructure.

`fed_statistics/` also runs against LUNG1 (no new node profile needed, since it works on any tabular CSV) — override `STATISTICS`/`FILTERS`/`OPTIONS` in `fed_statistics/run_study.py` to request different columns/statistics per run. `OPTIONS` controls privacy suppression (`suppress_threshold`, `suppress_secondary`/`suppress_key`) and quantile bootstrap iterations; see `algorithms/fed_statistics/test_stats_correctness.py` (run with `uv run pytest algorithms/fed_statistics/`) for worked examples of each statistic plus the secondary-suppression mechanism. If every node's data fails this validation (e.g. a typo'd column name), `central()` returns a clear `{"error": "No usable data across any organization"}` rather than silently reporting every statistic as unavailable.

`kaplan_meier/run_study.py` also exposes `NOISE_TYPE`/`SNR`/`RANDOM_SEED` — set `NOISE_TYPE` to `"gaussian"` (with `SNR > 0`) or `"poisson"` to perturb event times as an extra privacy layer before aggregation, on top of the existing `TIME_COL`/`EVENT_COL`/`STEP_DAYS` overrides. Leave `RANDOM_SEED` as `None` to let `central()` generate a fresh one per run (echoed back in the result for reproducibility), or set it to repeat a specific run's noise exactly. Both federated phases (`get_time_range`/`compute_events`) apply identical noise given the same seed. `uv run pytest algorithms/kaplan_meier/` covers both the core survival-curve math (`test_kaplan_meier.py` — cross-checked against an independent classical product-limit KM reference and against a single-organization pooled run, since the federated split/sum must reproduce the pooled result exactly) and the noise-injection mechanism (`test_noise_injection.py` — cross-phase consistency and backward-compatibility). Default `NOISE_TYPE = "none"` reproduces the pre-noise-injection curve exactly.

`logistic_regression/run_study.py` also exposes `N_ROUNDS`/`LOCAL_EPOCHS`/`LEARNING_RATE`/`TRAIN_TEST_RATIO`/`BATCH_RATIO`/`RANDOM_SEED` as real `central()` keyword arguments (forwarded into every `compute_stats`/`partial`/`evaluate` sub-task) — previously these were module globals read directly inside `central()`'s body, so changing them required editing `logistic_regression.py` and rebuilding the image; now they're overridable from `run_study.py` like every other algorithm's hyperparameters.

`average/`, `coxph/`, `kaplan_meier/`, and `logistic_regression/` now also raise a clear `ValueError` (listing the missing column name(s) and the dataset's available columns) instead of a raw `KeyError` when a configured column doesn't exist on a node's data — the same contract `fed_statistics/` already had via `_validate_column`/`_numeric_series`. This surfaces a typo'd column name in task logs immediately instead of as a buried pandas traceback.

## Adding your own algorithm

1. Create a new folder under `algorithms/` with an algorithm module and a `Dockerfile`.
2. Add any new dependencies to the root `pyproject.toml` and run `uv sync`.
3. Build the image: `docker build -t my-algo:latest algorithms/my-algo/`
4. Write a `run_study.py` that points at `my-algo:latest` and calls `"method": "central"`.

If the nodes report `non-existing Docker image`, see the root README's **Local image registry** section.
