"""Federated multiple linear regression via one-shot federated normal equations.

Data contract: `predictor_cols` (default PREDICTOR_COLS) must be numeric
columns, present under the same names on every node. `outcome_col` (default
OUTCOME_COL) must be a numeric, continuous column -- unlike
logistic_regression.py/coxph.py's binary outcome, there is no
representation layer here: there's no single canonical way to "derive" a
continuous value the way there is for a binary flag (presence/equals/isin),
so the column must already contain the real-valued outcome. A missing
column, or a column that isn't numeric after coercion, raises a clear
ValueError.

Unlike every other regression algorithm in this repo (logistic_regression.py's
FedAvg rounds, admm_logistic_regression.py's ADMM consensus rounds,
coxph.py's Newton-Raphson iterations), ordinary least squares needs no
iteration at all. Each node computes its local design matrix's X'X (p x p,
p = len(predictor_cols) + 1 for the intercept) and X'y (p x 1) sums, plus
the raw sum of y^2 (y'y). `central()` sums these three additive quantities
across nodes exactly once and solves the normal equations
`beta = (X'X)^-1 X'y` directly -- there's no `max_iterations`/`tolerance`
kwarg because there's nothing to converge: a single round produces the
exact pooled-data fit, not an approximation of it, and the whole
computation is fully deterministic (no `seed` kwarg either -- nothing here
is stochastic).

Because fitting is a single closed-form solve rather than an iterative
search, there's no `exp(X @ beta)` overflow risk the way there is in
coxph.py/logistic_regression.py, so predictor standardization isn't a
numerical necessity here the way it is there. Extremely different predictor
scales can still make X'X ill-conditioned (e.g. age in [0, 100] alongside a
lab value in the thousands); `central()` reports the condition number of
the pooled X'X as a warning if it's large, and returns a clear error
instead of garbage coefficients if X'X is outright singular (e.g. two
perfectly collinear predictors, or fewer usable rows than parameters once
every organization's data is combined).

`central()` also takes an `organizations` kwarg (default None): a list of
organization names to restrict the sub-task fan-out to, instead of every
organization in the collaboration. Without it, an organization that's
registered in the collaboration but has no node actually running gets a
sub-task it can never pick up, and client.wait_for_results() then blocks
forever waiting on it -- `organizations` lets the caller exclude it. An
unknown organization name raises a clear ValueError.

`central()` also takes `min_organizations` (default 1, i.e. no restriction)
and `min_rows` (default 0, i.e. no restriction). After phase 1
(`compute_local_sums`), any organization reporting fewer than `min_rows`
usable rows is dropped entirely from the fit -- excluded from X'X/X'y/y'y,
not just hidden from a count. If fewer than `min_organizations`
organizations remain after that filtering, `central()` returns a clear
error naming the organizations that fell short, instead of silently
fitting on whichever organizations happen to qualify -- having enough
organizations nominally included in the task is not the same as having
enough organizations with enough data. Like `organizations`, these are
requester-supplied kwargs (set in run_study.py), not a node-enforced
policy: they catch misconfiguration (an accidentally tiny test file, a
near-empty node), not a requester actively trying to bypass them.

`central()` also takes `max_pct_vars_vs_obs` (default None, i.e. no
restriction): if set, `central()` errors out after phase 1 instead of
fitting when the number of parameters (predictors plus the intercept) is
more than this percentage of total usable rows (e.g. 10 means at least 10
rows per parameter required) -- a basic overfitting/memorization guard,
checked against the row total after `min_rows` exclusion has already
dropped under-qualified organizations. Separately, `central()` always
refuses to fit (regardless of `max_pct_vars_vs_obs`) if there are not
enough residual degrees of freedom left to estimate the residual variance
at all (total usable rows <= number of parameters).
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import f, t
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.tools.util import error, info, warn

PREDICTOR_COLS = ["age", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage"]
OUTCOME_COL = "Survival.time"

MIN_ORGANIZATIONS = 1
MIN_ROWS = 0
MAX_PCT_VARS_VS_OBS = None


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


def _build_results_table(beta: np.ndarray, se: np.ndarray, var_names: list, df_resid: int) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        tvalues = np.divide(
            beta,
            se,
            out=np.zeros_like(beta, dtype=float),
            where=se > 0,
        )
    pvalues = 2.0 * t.sf(np.abs(tvalues), df_resid) if df_resid > 0 else np.full_like(beta, np.nan)
    crit = float(t.ppf(0.975, df_resid)) if df_resid > 0 else float("nan")

    results = pd.DataFrame(
        {
            "Coef": np.around(beta, 5),
            "SE": np.around(se, 5),
            "Var": var_names,
            "t": np.around(tvalues, 5),
            "p-value": pvalues,
        }
    )
    results["lower_CI"] = np.around(results["Coef"] - crit * results["SE"], 5)
    results["upper_CI"] = np.around(results["Coef"] + crit * results["SE"], 5)
    return results.set_index("Var")


@data(1)
@algorithm_client
def compute_local_sums(
    client: AlgorithmClient,
    df: pd.DataFrame,
    predictor_cols: list,
    outcome_col: str,
) -> dict:
    """Local design-matrix sums for one node's contribution to the pooled
    normal equations. organization_id is reported so central() can tell
    which organization a given row count belongs to, to drop one below
    min_rows from the fit entirely, not just hide it from a count."""
    _require_columns(df, [*predictor_cols, outcome_col], "compute_local_sums")
    n_params = len(predictor_cols) + 1
    empty = {
        "n": 0,
        "XtX": np.zeros((n_params, n_params)).tolist(),
        "XtY": [0.0] * n_params,
        "YtY": 0.0,
        "organization_id": client.organization_id,
    }

    numeric_cols = [*predictor_cols, outcome_col]
    try:
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        msg = f"compute_local_sums: column(s) {numeric_cols} must all be numeric ({exc})"
        error(msg)
        raise ValueError(msg) from exc

    clean = numeric_df.dropna()
    info(f"Dataset: {len(df)} total rows, {len(clean)} usable rows after dropna")
    if len(clean) == 0:
        return empty

    X = np.column_stack([np.ones(len(clean)), clean[predictor_cols].to_numpy(dtype=float)])
    y = clean[outcome_col].to_numpy(dtype=float)

    XtX = X.T @ X
    XtY = X.T @ y
    YtY = float(np.dot(y, y))

    return {
        "n": len(clean),
        "XtX": XtX.tolist(),
        "XtY": XtY.tolist(),
        "YtY": YtY,
        "organization_id": client.organization_id,
    }


@algorithm_client
def central(
    client: AlgorithmClient,
    predictor_cols: list = None,
    outcome_col: str = OUTCOME_COL,
    organizations: list | None = None,
    min_organizations: int = MIN_ORGANIZATIONS,
    min_rows: int = MIN_ROWS,
    max_pct_vars_vs_obs: float | None = MAX_PCT_VARS_VS_OBS,
) -> dict:
    predictor_cols = list(predictor_cols) if predictor_cols else list(PREDICTOR_COLS)
    n_params = len(predictor_cols) + 1  # +1 for the intercept

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
    info("FEDERATED LINEAR REGRESSION")
    info("=" * 60)
    info(f"Organizations : {[org['name'] for org in orgs]}")
    info(f"Predictors    : {predictor_cols}")
    info(f"Outcome       : {outcome_col}")

    # ── Phase 1: local design-matrix sums ────────────────────────────────────
    info("")
    info("── PHASE 1: Local design-matrix sums ───────────────────────")
    task = client.task.create(
        input_={
            "method": "compute_local_sums",
            "kwargs": {"predictor_cols": predictor_cols, "outcome_col": outcome_col},
        },
        organizations=org_ids,
        name="linreg_local_sums",
        description="Compute local design-matrix sums for federated OLS",
    )
    results = client.wait_for_results(task["id"])

    # Drop any organization below min_rows entirely before anything else is
    # computed -- it must not contribute to X'X/X'y/y'y, not just be
    # excluded from the row count.
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
                "leave no residual degrees of freedom to estimate the residual "
                "variance from."
            )
        }

    XtX_sum = np.zeros((n_params, n_params), dtype=float)
    XtY_sum = np.zeros(n_params, dtype=float)
    YtY_sum = 0.0
    for r in qualifying:
        XtX_sum += np.array(r["XtX"], dtype=float)
        XtY_sum += np.array(r["XtY"], dtype=float)
        YtY_sum += float(r["YtY"])

    # scipy.linalg.solve only raises for an *exactly* singular matrix, which
    # floating-point noise makes vanishingly rare even for predictors that
    # are perfectly collinear in theory (it emits a warning and returns a
    # numerically unstable answer instead) -- rank is the correct check for
    # "this system has no unique solution", not whether solve() happened to
    # raise.
    rank = int(np.linalg.matrix_rank(XtX_sum))
    if rank < n_params:
        return {
            "error": (
                f"The pooled design matrix is rank-deficient (rank={rank}, expected "
                f"{n_params}) -- likely two or more predictors are perfectly (or "
                "near-perfectly) collinear once every organization's data is "
                "combined, or there are fewer usable rows than parameters. Refusing "
                "to fit, since the normal equations have no unique solution."
            )
        }

    try:
        beta = solve(XtX_sum, XtY_sum)
    except np.linalg.LinAlgError:
        return {
            "error": (
                "The pooled design matrix is singular -- likely two or more "
                "predictors are perfectly collinear once every organization's data "
                "is combined, or there are fewer usable rows than parameters. "
                "Refusing to fit with an ill-defined solution."
            )
        }

    cond_number = float(np.linalg.cond(XtX_sum))

    # Residual sum of squares via the standard OLS identity RSS = y'y - beta'X'y
    # -- avoids ever centralizing raw rows or residuals, only the three
    # additive sums every node already reported.
    rss = max(float(YtY_sum - beta @ XtY_sum), 0.0)
    # The intercept column is all 1s, so (X'y)[0] == sum(y) already -- no
    # separate y_sum needs to be reported.
    y_sum = float(XtY_sum[0])
    tss = max(float(YtY_sum - (y_sum**2) / total_n), 0.0)

    sigma2 = rss / df_resid
    covariance = sigma2 * _safe_inverse(XtX_sum)
    se = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))

    r_squared = float(1.0 - rss / tss) if tss > 0 else float("nan")
    n_predictors = n_params - 1
    if tss > 0:
        adjusted_r_squared = float(1.0 - (1.0 - r_squared) * (total_n - 1) / df_resid)
    else:
        adjusted_r_squared = float("nan")

    if n_predictors > 0 and tss > 0:
        f_statistic = float("inf") if rss == 0 else ((tss - rss) / n_predictors) / (rss / df_resid)
        overall_p_value = float(f.sf(f_statistic, n_predictors, df_resid))
    else:
        f_statistic = float("nan")
        overall_p_value = float("nan")

    if rss > 0:
        log_likelihood = -0.5 * total_n * (np.log(2 * np.pi) + np.log(rss / total_n) + 1)
        aic = float(-2.0 * log_likelihood + 2.0 * (n_params + 1))
    else:
        aic = float("nan")

    results_table = _build_results_table(beta, se, ["Intercept", *predictor_cols], df_resid)

    model_warnings = []
    if cond_number > 1e10 or np.isnan(cond_number) or np.isinf(cond_number):
        msg = (
            f"Design matrix is poorly conditioned (condition number={cond_number:.2e}); "
            "coefficients and standard errors may be unreliable. This usually means "
            "two or more predictors are highly correlated (or nearly constant) once "
            "every organization's data is combined."
        )
        warn(msg)
        model_warnings.append(msg)

    info(f"Final coefficients: {beta.tolist()}")
    return {
        "included_organizations": org_ids,
        "excluded_organizations": excluded_ids,
        "model": results_table.to_json(),
        "overall_p_value": overall_p_value,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "aic": aic,
        "degrees_of_freedom": df_resid,
        "num_observations": total_n,
        "warnings": model_warnings,
    }
