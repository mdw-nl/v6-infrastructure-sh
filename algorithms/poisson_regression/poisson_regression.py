"""Federated Poisson regression via iteratively reweighted least squares (IRLS).

Data contract: `predictor_cols` (default PREDICTOR_COLS) must be numeric
columns, present under the same names on every node. `outcome_col`
(default OUTCOME_COL) must be a non-negative, (effectively) integer-valued
count column -- not a representation-derived binary flag like
logistic_regression.py/coxph.py's outcome, and not a continuous value like
linear_regression.py's. A negative value, a non-integer value, or a column
that isn't numeric at all raises a clear ValueError. A duplicate name
within `predictor_cols`, or `outcome_col` also appearing in
`predictor_cols`, raises a clear ValueError too -- rather than silently
building a design matrix with more columns than `predictor_cols` has
entries, which is what a duplicate label otherwise does once it goes
through pandas' column selection.

Unlike linear_regression.py's one-shot normal-equations solve, Poisson
regression's log link (`log(mu) = X @ beta`) and its variance-equals-mean
relationship (`Var(y) = mu`, the defining property of the Poisson
distribution) mean the per-row weights and adjusted response used in each
round's weighted least-squares step depend on the *current* beta estimate
-- so, like coxph.py's Newton-Raphson, this needs genuine iteration, not a
single aggregation round. Each round, every node computes `eta = X @ beta`,
`mu = exp(eta)`, weight `W = mu`, and adjusted response
`z = eta + (y - mu) / mu`, then reports `X'WX` and `X'Wz` (plus that
round's deviance contribution); `central()` sums these across nodes and
solves `beta_new = (X'WX)^-1 X'Wz`, repeating until `max(|beta_new -
beta_old|) <= tolerance` or `max_iterations` is reached (the same
convergence style as coxph.py's Newton-Raphson, not a different one).

Phase 1 (`compute_local_row_stats`) runs once, before any IRLS round,
purely to learn each node's usable row count (for `min_rows` filtering,
before any node contributes to a weighted sum) and three fixed,
beta-independent quantities needed later for the null-model deviance and
the saturated log-likelihood (`sum(y)`, `sum(y*log(y))` with `y*log(y)
:= 0` at `y=0`, and `sum(log(y!))` via `scipy.special.gammaln`) -- these
let `central()` compute the null-model (intercept-only) deviance in closed
form (the Poisson intercept-only MLE is exactly `log(mean(y))`, so no
iteration is needed for the null model), and from there a proper
likelihood-ratio test of the fitted model against it (`overall_p_value`)
and AIC, all from additive per-node sums -- no raw row ever needs to leave
a node for any of this.

Because the log link means `exp(X @ beta)` can overflow the same way it
can in coxph.py, `eta` is clipped to [-30, 30] before exponentiating in
every round -- a simpler guard than coxph.py's full covariate
standardization (pooled centre/scale, back-transform at the end), chosen
to keep this algorithm's structure as close to linear_regression.py's as
the iteration requirement allows. Predictors on very different scales may
still need caller-side rescaling for good numerical conditioning; a large
condition number on the final `X'WX` is reported as a warning rather than
silently producing unreliable standard errors.

`central()` reports a `dispersion_estimate` (Pearson chi-squared statistic
over residual degrees of freedom) alongside the fit -- a value well above
1 means the data is more variable than the Poisson distribution's
`Var(y) = mu` assumption allows (overdispersion), a common real-world
violation that makes the reported standard errors too small; `central()`
warns about this rather than silently reporting falsely-confident p-values.

Unlike coxph.py, no node-side method here verifies it was dispatched by
central() as a proper sub-task: coxph's guard exists because its
iteratively-released risk-set aggregates have a specific leak (two
releases close together can isolate one individual's exact covariate
vector by subtraction). Each round here releases one ordinary weighted
sum per organization, structurally the same as
admm_logistic_regression.py's iterative rounds (which also has no such
guard) -- there is no equivalent differencing leak to guard against.

`central()` also takes an `organizations` kwarg (default None): a list of
organization names to restrict the sub-task fan-out to, instead of every
organization in the collaboration. Without it, an organization that's
registered in the collaboration but has no node actually running gets a
sub-task it can never pick up, and client.wait_for_results() then blocks
forever waiting on it -- `organizations` lets the caller exclude it. An
unknown organization name raises a clear ValueError.

`central()` also takes `min_organizations` (default 1, i.e. no restriction)
and `min_rows` (default 0, i.e. no restriction). After phase 1
(`compute_local_row_stats`), any organization reporting fewer than
`min_rows` usable rows is dropped entirely from the fit -- excluded from
every IRLS round's weighted sums, not just hidden from a count. If fewer
than `min_organizations` organizations remain after that filtering,
`central()` returns a clear error naming the organizations that fell
short, instead of silently fitting on whichever organizations happen to
qualify. Like `organizations`, these are requester-supplied kwargs (set in
run_study.py), not a node-enforced policy.

`central()` also takes `max_pct_vars_vs_obs` (default None, i.e. no
restriction): if set, `central()` errors out after phase 1 instead of
fitting when the number of parameters (predictors plus the intercept) is
more than this percentage of total usable rows, checked against the row
total after `min_rows` exclusion has already dropped under-qualified
organizations.

`central()` also takes `exposure_col` (default None, i.e. no offset):
a numeric, strictly positive column (e.g. person-years of follow-up,
population size) representing how much opportunity each row had to
accumulate `outcome_col` events. When set, the model becomes
`log(mu) = log(exposure) + X @ beta` instead of `log(mu) = X @ beta` --
`log(exposure)` enters the linear predictor with a fixed coefficient of
1, not a coefficient `central()` fits, which is what makes it an offset
rather than another predictor. This turns the model from predicting raw
counts into predicting a *rate* (events per unit of exposure): two rows
with identical `predictor_cols` but different exposure are now expected
to produce proportionally different counts, rather than the same count.
Every per-round working-response/weight calculation accounts for the
offset (see `compute_local_glm_sums`), and the null-model deviance and
starting intercept are both derived from the exposure-weighted rate
`sum(y) / sum(exposure)` rather than the plain mean of `y` -- when
`exposure_col` is left at its default, `exposure` is implicitly 1 for
every row, `log(exposure) = 0`, and both of those reduce exactly to the
no-offset formulas above (this is a strict extension, not a separate code
path). A non-positive exposure value raises a clear ValueError, since
`log(exposure)` would otherwise be undefined or `-inf`; a zero-exposure
row (someone with no opportunity to have an event) should be excluded by
the caller before running, not passed in.
"""

import math

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.special import gammaln
from scipy.stats import chi2, norm
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.tools.util import error, info, warn

PREDICTOR_COLS = ["age", "clinical.T.Stage", "Clinical.M.Stage"]
OUTCOME_COL = "Clinical.N.Stage"
EXPOSURE_COL = None

MAX_ITERATIONS = 25
TOLERANCE = 1e-8

MIN_ORGANIZATIONS = 1
MIN_ROWS = 0
MAX_PCT_VARS_VS_OBS = None

ETA_CLIP = 30.0
DISPERSION_WARNING_THRESHOLD = 1.5


def _require_columns(df: pd.DataFrame, columns: list, method_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        msg = (
            f"{method_name}: column(s) {missing} not found in dataset; "
            f"available columns: {list(df.columns)}"
        )
        error(msg)
        raise ValueError(msg)


def _filter_organizations(orgs: list, organizations: list | None, method_name: str) -> list:
    """Restrict `orgs` (from client.organization.list()) to the ones named in
    `organizations`, or return `orgs` unchanged if `organizations` is None
    (the default: every organization in the collaboration). Filtering by name
    lets a caller exclude an organization that's registered in the
    collaboration but has no node actually running -- fanning a sub-task out
    to one hangs client.wait_for_results() forever, since it waits for a
    result from every organization a task was sent to."""
    if organizations is None:
        return orgs
    if not organizations:
        msg = (
            f"{method_name}: 'organizations' was given as an empty list; omit it "
            f"(or pass None) to use every organization in the collaboration, or "
            f"list at least one organization name"
        )
        error(msg)
        raise ValueError(msg)
    available = {org["name"] for org in orgs}
    missing = [name for name in organizations if name not in available]
    if missing:
        msg = (
            f"{method_name}: organization(s) {missing} not found in the collaboration; "
            f"available organizations: {sorted(available)}"
        )
        error(msg)
        raise ValueError(msg)
    selected = set(organizations)
    return [org for org in orgs if org["name"] in selected]


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        warn("Matrix inversion failed; using pseudo-inverse")
        return np.linalg.pinv(matrix)


def _clean_and_validate(
    df: pd.DataFrame,
    predictor_cols: list,
    outcome_col: str,
    method_name: str,
    exposure_col: str | None = None,
) -> pd.DataFrame:
    """Coerce predictor_cols + outcome_col (+ exposure_col, if given) to
    numeric, drop unusable rows, and confirm outcome_col is a non-negative,
    (effectively) integer count column and exposure_col (if given) is
    strictly positive -- shared by every phase so the usable row set is
    identical across all of them."""
    columns_needed = [*predictor_cols, outcome_col]
    if exposure_col is not None:
        columns_needed = [*columns_needed, exposure_col]
    _require_columns(df, columns_needed, method_name)
    try:
        numeric_df = df[columns_needed].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        msg = f"{method_name}: column(s) {columns_needed} must all be numeric ({exc})"
        error(msg)
        raise ValueError(msg) from exc

    clean = numeric_df.dropna()
    y = clean[outcome_col]
    if (y < 0).any():
        msg = (
            f"{method_name}: outcome_col {outcome_col!r} contains negative value(s); "
            "Poisson regression requires a non-negative count outcome"
        )
        error(msg)
        raise ValueError(msg)
    if not np.allclose(y, np.round(y)):
        msg = (
            f"{method_name}: outcome_col {outcome_col!r} contains non-integer value(s); "
            "Poisson regression models count data, which must be whole numbers"
        )
        error(msg)
        raise ValueError(msg)
    if exposure_col is not None and (clean[exposure_col] <= 0).any():
        msg = (
            f"{method_name}: exposure_col {exposure_col!r} contains non-positive value(s); "
            "exposure must be strictly positive (it's used as log(exposure) in the "
            "offset). Exclude zero-exposure rows before running, rather than passing "
            "them in."
        )
        error(msg)
        raise ValueError(msg)
    return clean


def _build_results_table(beta: np.ndarray, se: np.ndarray, var_names: list) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        zvalues = np.divide(
            beta,
            se,
            out=np.zeros_like(beta, dtype=float),
            where=se > 0,
        )
    pvalues = 2.0 * norm.sf(np.abs(zvalues))

    results = pd.DataFrame(
        {
            "Coef": np.around(beta, 5),
            "Exp(coef)": np.around(np.exp(beta), 5),
            "SE": np.around(se, 5),
            "Var": var_names,
            "Z": zvalues,
            "p-value": pvalues,
        }
    )
    results["lower_CI"] = np.around(np.exp(results["Coef"] - 1.96 * results["SE"]), 5)
    results["upper_CI"] = np.around(np.exp(results["Coef"] + 1.96 * results["SE"]), 5)
    return results.set_index("Var")


@data(1)
@algorithm_client
def compute_local_row_stats(
    client: AlgorithmClient,
    df: pd.DataFrame,
    predictor_cols: list,
    outcome_col: str,
    exposure_col: str | None = None,
) -> dict:
    """Usable row count plus fixed, beta-independent sums needed for the
    null-model deviance and saturated log-likelihood -- see the module
    docstring. `exposure_sum`/`y_log_exposure_sum` are the offset-aware
    additions: with no exposure_col, exposure is implicitly 1 for every row
    (log(exposure) = 0), so exposure_sum reduces to n and
    y_log_exposure_sum reduces to 0, matching the no-offset formulas
    exactly. organization_id is reported so central() can tell which
    organization a given row count belongs to, to drop one below min_rows
    from the fit entirely, not just hide it from a count."""
    empty = {
        "n": 0,
        "y_sum": 0.0,
        "y_log_y_sum": 0.0,
        "log_factorial_sum": 0.0,
        "exposure_sum": 0.0,
        "y_log_exposure_sum": 0.0,
        "organization_id": client.organization_id,
    }
    clean = _clean_and_validate(df, predictor_cols, outcome_col, "compute_local_row_stats", exposure_col)
    info(f"Dataset: {len(df)} total rows, {len(clean)} usable rows after dropna")
    if len(clean) == 0:
        return empty

    y = clean[outcome_col].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_log_y = np.where(y > 0, y * np.log(y), 0.0)

    if exposure_col is not None:
        exposure = clean[exposure_col].to_numpy(dtype=float)
    else:
        exposure = np.ones(len(clean), dtype=float)
    log_exposure = np.log(exposure)

    return {
        "n": len(clean),
        "y_sum": float(y.sum()),
        "y_log_y_sum": float(y_log_y.sum()),
        "log_factorial_sum": float(gammaln(y + 1).sum()),
        "exposure_sum": float(exposure.sum()),
        "y_log_exposure_sum": float((y * log_exposure).sum()),
        "organization_id": client.organization_id,
    }


@data(1)
@algorithm_client
def compute_local_glm_sums(
    client: AlgorithmClient,
    df: pd.DataFrame,
    predictor_cols: list,
    outcome_col: str,
    beta: list,
    exposure_col: str | None = None,
) -> dict:
    """One IRLS round's weighted design-matrix sums for one node, given the
    current global beta. With exposure_col set, eta includes the fixed
    offset log(exposure) before the weight/working-response calculations,
    and the offset's (fixed, non-fitted) contribution is subtracted back
    out of the working response before it's regressed against X -- X'WX/
    X'Wz describe beta's contribution only, never the offset's. Also
    returns this round's deviance and Pearson residual sum of squares --
    cheap to compute alongside X'WX/X'Wz, and central() only keeps the
    final round's values once IRLS has converged."""
    clean = _clean_and_validate(df, predictor_cols, outcome_col, "compute_local_glm_sums", exposure_col)
    n_params = len(predictor_cols) + 1
    empty = {
        "n": 0,
        "XtWX": np.zeros((n_params, n_params)).tolist(),
        "XtWz": [0.0] * n_params,
        "deviance": 0.0,
        "pearson_ssq": 0.0,
        "organization_id": client.organization_id,
    }
    if len(clean) == 0:
        return empty

    X = np.column_stack([np.ones(len(clean)), clean[predictor_cols].to_numpy(dtype=float)])
    y = clean[outcome_col].to_numpy(dtype=float)

    if exposure_col is not None:
        exposure = clean[exposure_col].to_numpy(dtype=float)
    else:
        exposure = np.ones(len(clean), dtype=float)
    offset = np.log(exposure)

    beta_arr = np.array(beta, dtype=float)
    eta = np.clip(offset + X @ beta_arr, -ETA_CLIP, ETA_CLIP)
    mu = np.exp(eta)
    mu_safe = np.clip(mu, 1e-10, None)

    W = mu_safe
    z = eta + (y - mu_safe) / mu_safe
    z_adjusted = z - offset  # offset's contribution is fixed, not fitted -- regress only on X

    XtWX = (X * W[:, None]).T @ X
    XtWz = (X * W[:, None]).T @ z_adjusted

    with np.errstate(divide="ignore", invalid="ignore"):
        y_log_ratio = np.where(y > 0, y * np.log(y / mu_safe), 0.0)
    deviance = float(2.0 * np.sum(y_log_ratio - (y - mu_safe)))
    pearson_ssq = float(np.sum((y - mu_safe) ** 2 / mu_safe))

    return {
        "n": len(clean),
        "XtWX": XtWX.tolist(),
        "XtWz": XtWz.tolist(),
        "deviance": deviance,
        "pearson_ssq": pearson_ssq,
        "organization_id": client.organization_id,
    }


@algorithm_client
def central(
    client: AlgorithmClient,
    predictor_cols: list = None,
    outcome_col: str = OUTCOME_COL,
    exposure_col: str | None = EXPOSURE_COL,
    organizations: list | None = None,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
    min_organizations: int = MIN_ORGANIZATIONS,
    min_rows: int = MIN_ROWS,
    max_pct_vars_vs_obs: float | None = MAX_PCT_VARS_VS_OBS,
) -> dict:
    predictor_cols = list(predictor_cols) if predictor_cols else list(PREDICTOR_COLS)
    n_params = len(predictor_cols) + 1  # +1 for the intercept
    max_iterations = max(1, int(max_iterations))
    tolerance = float(tolerance)

    if len(set(predictor_cols)) != len(predictor_cols):
        # pandas silently multiplies columns out (not just duplicates them)
        # when a frame that already has a repeated label is indexed with
        # that same label again, so a duplicate name here would otherwise
        # surface many rounds later as a confusing dimension/rank mismatch
        # instead of a clear error naming the actual mistake.
        duplicates = sorted({c for c in predictor_cols if predictor_cols.count(c) > 1})
        return {"error": f"'predictor_cols' contains duplicate column name(s): {duplicates}"}
    if outcome_col in predictor_cols:
        return {
            "error": f"'outcome_col' ({outcome_col!r}) must not also appear in 'predictor_cols'"
        }
    if exposure_col is not None:
        if exposure_col in predictor_cols:
            return {
                "error": f"'exposure_col' ({exposure_col!r}) must not also appear in 'predictor_cols'"
            }
        if exposure_col == outcome_col:
            return {"error": f"'exposure_col' must not be the same column as 'outcome_col' ({outcome_col!r})"}

    min_organizations = int(min_organizations)
    if min_organizations < 1:
        return {"error": f"'min_organizations' must be at least 1, got {min_organizations}"}
    if min_rows < 0:
        return {"error": f"'min_rows' must be >= 0, got {min_rows}"}
    if max_pct_vars_vs_obs is not None and max_pct_vars_vs_obs <= 0:
        return {
            "error": f"'max_pct_vars_vs_obs' must be positive if set, got {max_pct_vars_vs_obs}"
        }

    orgs = _filter_organizations(client.organization.list(), organizations, "central")
    org_ids = [org["id"] for org in orgs]
    org_name_by_id = {org["id"]: org["name"] for org in orgs}

    if len(org_ids) < min_organizations:
        return {
            "error": (
                f"Only {len(org_ids)} organization(s) are included in this task, but "
                f"'min_organizations' requires at least {min_organizations}."
            )
        }

    info("=" * 60)
    info("FEDERATED POISSON REGRESSION")
    info("=" * 60)
    info(f"Organizations : {[org['name'] for org in orgs]}")
    info(f"Predictors    : {predictor_cols}")
    info(f"Outcome       : {outcome_col}")
    info(f"Exposure      : {exposure_col if exposure_col else 'none (modeling raw counts)'}")

    # ── Phase 1: local row stats ─────────────────────────────────────────────
    info("")
    info("── PHASE 1: Local row stats ─────────────────────────────────")
    task = client.task.create(
        input_={
            "method": "compute_local_row_stats",
            "kwargs": {
                "predictor_cols": predictor_cols,
                "outcome_col": outcome_col,
                "exposure_col": exposure_col,
            },
        },
        organizations=org_ids,
        name="poisson_row_stats",
        description="Compute local row counts and fixed sums for federated Poisson regression",
    )
    results = client.wait_for_results(task["id"])

    qualifying = [r for r in results if r["n"] >= min_rows]
    qualifying_ids = [r["organization_id"] for r in qualifying]
    if len(qualifying_ids) < min_organizations:
        below_threshold = [
            org_name_by_id.get(r["organization_id"], r["organization_id"])
            for r in results
            if r["n"] < min_rows
        ]
        return {
            "error": (
                f"Only {len(qualifying_ids)} organization(s) have at least "
                f"'min_rows'={min_rows} usable rows, but 'min_organizations' requires "
                f"at least {min_organizations}. Organization(s) below the row "
                f"threshold: {below_threshold}."
            )
        }
    excluded_ids = [r["organization_id"] for r in results if r["organization_id"] not in qualifying_ids]
    org_ids = qualifying_ids

    total_n = sum(r["n"] for r in qualifying)
    info(f"Included organizations: {org_ids} (excluded: {excluded_ids})")
    info(f"Total usable rows: {total_n}")
    if total_n == 0:
        return {"error": "No usable data across any organization"}

    if max_pct_vars_vs_obs is not None:
        pct_vars_vs_obs = n_params * 100 / total_n
        if pct_vars_vs_obs > max_pct_vars_vs_obs:
            return {
                "error": (
                    f"Number of parameters ({n_params}, including the intercept) is "
                    f"{pct_vars_vs_obs:.1f}% of total usable rows ({total_n}), which "
                    f"exceeds 'max_pct_vars_vs_obs'={max_pct_vars_vs_obs}%. Refusing to "
                    "fit, as this risks a model that memorizes individual records "
                    "instead of learning a general pattern."
                )
            }

    df_resid = total_n - n_params
    if df_resid <= 0:
        return {
            "error": (
                f"Not enough data to fit: {total_n} usable row(s) across qualifying "
                f"organizations, but {n_params} parameters (including the intercept) "
                "leave no residual degrees of freedom."
            )
        }

    y_sum_total = sum(r["y_sum"] for r in qualifying)
    y_log_y_sum_total = sum(r["y_log_y_sum"] for r in qualifying)
    log_factorial_sum_total = sum(r["log_factorial_sum"] for r in qualifying)
    exposure_sum_total = sum(r["exposure_sum"] for r in qualifying)
    y_log_exposure_sum_total = sum(r["y_log_exposure_sum"] for r in qualifying)

    # With no exposure_col, exposure_sum_total == total_n and
    # y_log_exposure_sum_total == 0 (see compute_local_row_stats), so
    # global_rate reduces exactly to mean(y) and the two lines below
    # reduce exactly to the no-offset formulas -- this is one derivation,
    # not a separate offset-vs-no-offset code path. The intercept-only
    # (plus fixed offset) Poisson MLE is exactly log(sum(y)/sum(exposure)),
    # so the null model's deviance is available in closed form -- no
    # iteration needed for it. See the module docstring for the full
    # derivation.
    global_rate = y_sum_total / exposure_sum_total
    safe_rate = max(global_rate, 1e-6)
    deviance_null = 2.0 * (
        y_log_y_sum_total - y_log_exposure_sum_total - y_sum_total * math.log(safe_rate)
    )
    saturated_loglik = y_log_y_sum_total - y_sum_total - log_factorial_sum_total

    # ── Phase 2: IRLS iterations ─────────────────────────────────────────────
    info("")
    info("── PHASE 2: IRLS iterations ──────────────────────────────────")
    beta = np.zeros(n_params, dtype=float)
    beta[0] = math.log(safe_rate)

    deviance_final = deviance_null
    pearson_ssq_final = 0.0
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        task = client.task.create(
            input_={
                "method": "compute_local_glm_sums",
                "kwargs": {
                    "predictor_cols": predictor_cols,
                    "outcome_col": outcome_col,
                    "beta": beta.tolist(),
                    "exposure_col": exposure_col,
                },
            },
            organizations=org_ids,
            name=f"poisson_irls_round_{iteration}",
            description=f"Local weighted design-matrix sums, iteration {iteration}",
        )
        round_results = client.wait_for_results(task["id"])

        XtWX_sum = np.zeros((n_params, n_params), dtype=float)
        XtWz_sum = np.zeros(n_params, dtype=float)
        deviance_final = 0.0
        pearson_ssq_final = 0.0
        for r in round_results:
            XtWX_sum += np.array(r["XtWX"], dtype=float)
            XtWz_sum += np.array(r["XtWz"], dtype=float)
            deviance_final += r["deviance"]
            pearson_ssq_final += r["pearson_ssq"]

        rank = int(np.linalg.matrix_rank(XtWX_sum))
        if rank < n_params:
            return {
                "error": (
                    f"The pooled weighted design matrix is rank-deficient (rank={rank}, "
                    f"expected {n_params}) -- likely two or more predictors are "
                    "perfectly (or near-perfectly) collinear once every organization's "
                    "data is combined. Refusing to fit."
                )
            }

        try:
            beta_new = solve(XtWX_sum, XtWz_sum)
        except np.linalg.LinAlgError:
            return {
                "error": (
                    "The pooled weighted design matrix is singular; refusing to fit "
                    "with an ill-defined solution."
                )
            }

        delta = float(np.max(np.abs(beta_new - beta)))
        beta = beta_new
        info(f"Iteration {iteration}: max |Δbeta| = {delta:.3e}, deviance = {deviance_final:.4f}")

        if math.isnan(delta) or math.isinf(delta):
            return {
                "error": (
                    "IRLS produced a NaN/Inf update. This usually means extreme "
                    "predictor scales or a near-perfect fit; consider rescaling "
                    "predictor_cols."
                )
            }
        if delta <= tolerance:
            info("Betas have settled; IRLS converged")
            converged = True
            break

    model_warnings = []
    if not converged:
        msg = f"IRLS did not converge within {max_iterations} iteration(s); results may be unreliable."
        warn(msg)
        model_warnings.append(msg)

    covariance = _safe_inverse(XtWX_sum)
    se = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))

    cond_number = float(np.linalg.cond(XtWX_sum))
    if cond_number > 1e10 or np.isnan(cond_number) or np.isinf(cond_number):
        msg = (
            f"Design matrix is poorly conditioned (condition number={cond_number:.2e}); "
            "coefficients and standard errors may be unreliable. This usually means "
            "two or more predictors are highly correlated (or nearly constant) once "
            "every organization's data is combined."
        )
        warn(msg)
        model_warnings.append(msg)

    dispersion_estimate = pearson_ssq_final / df_resid if df_resid > 0 else float("nan")
    if dispersion_estimate > DISPERSION_WARNING_THRESHOLD:
        msg = (
            f"Estimated dispersion ({dispersion_estimate:.2f}) is well above 1, "
            "suggesting the data is overdispersed relative to the Poisson "
            "assumption (Var(y) == mean(y)). Standard errors reported here are "
            "likely too small; consider a quasi-Poisson or negative-binomial model."
        )
        warn(msg)
        model_warnings.append(msg)

    n_predictors = n_params - 1
    loglik_fitted = saturated_loglik - deviance_final / 2.0
    aic = float(-2.0 * loglik_fitted + 2.0 * n_params)

    if n_predictors > 0:
        lr_statistic = max(deviance_null - deviance_final, 0.0)
        overall_p_value = float(chi2.sf(lr_statistic, n_predictors))
    else:
        overall_p_value = float("nan")

    results_table = _build_results_table(beta, se, ["Intercept", *predictor_cols])

    info(f"Final coefficients: {beta.tolist()}")
    return {
        "included_organizations": org_ids,
        "excluded_organizations": excluded_ids,
        "model": results_table.to_json(),
        "converged": converged,
        "iterations": iteration,
        "deviance": deviance_final,
        "null_deviance": deviance_null,
        "overall_p_value": overall_p_value,
        "aic": aic,
        "degrees_of_freedom": df_resid,
        "num_observations": total_n,
        "dispersion_estimate": dispersion_estimate,
        "warnings": model_warnings,
    }
