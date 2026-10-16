# LUNG1 data

Four CSVs, one per demo organization (`alpha`/`beta`/`gamma`/`theta`), used by
`infrastructure/nodes.lung1.env` — the harness's default node profile (no `--nodes` flag
needed). No generator script produces these; they're committed as static files.

**Provenance: hand-crafted synthetic data.** These rows were manually created to mimic a
LUNG1-like schema for demo/testing purposes. They are **not** a real patient dataset —
don't draw clinical conclusions from them, and don't assume the row-level statistics
reflect any real population.

## Columns

| Column | Type | Notes |
|---|---|---|
| `PatientID` | string | e.g. `LUNG1-001` — not used as a feature by any algorithm |
| `age` | float | |
| `clinical.T.Stage` | int | |
| `Clinical.N.Stage` | int | |
| `Clinical.M.Stage` | int | |
| `Overall.Stage` | string | e.g. `I`, `IIIb` — not used as a feature by any algorithm's defaults |
| `Histology` | string | e.g. `large cell`, `squamous cell carcinoma` — not used by any algorithm's defaults |
| `gender` | string | not used by any algorithm's defaults |
| `Survival.time` | int | days; used as the time-to-event column by `coxph`/`kaplan_meier` |
| `deadstatus.event` | int (0/1) | binary target used by `logistic_regression`/`admm_logistic_regression`, and the event indicator for `coxph`/`kaplan_meier` |
| `survival_1y` | int (0/1) | **known leaky column** — confirmed ~99.4% correlated with `deadstatus.event` (it's a near-deterministic function of `Survival.time` >= 365 days). Never use it as a feature; `logistic_regression`'s test suite (`test_default_feature_cols_do_not_contain_the_known_leaky_column`) guards against it being reintroduced by accident. |

## Row counts

| Org | Rows (excl. header) |
|---|---|
| alpha | 133 |
| beta | 133 |
| gamma | 133 |
| theta | 105 |

## Default usage across algorithms

| Algorithm | Default columns used |
|---|---|
| `average/` | `age` |
| `logistic_regression/`, `admm_logistic_regression/` | features: `age`/`clinical.T.Stage`/`Clinical.N.Stage`/`Clinical.M.Stage`; target: `deadstatus.event` |
| `coxph/` | time: `Survival.time`; event: `deadstatus.event`; covariates: `age`/`clinical.T.Stage`/`Clinical.N.Stage`/`Clinical.M.Stage` |
| `kaplan_meier/` | time: `Survival.time`; event: `deadstatus.event` |
| `fed_statistics/` | `age`, `gender`, `Overall.Stage`, `Survival.time` |
