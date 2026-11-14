"""Federated Cox proportional hazards regression via Newton-Raphson iterations.

Data contract: `time_col` (default TIME_COL) must be a non-negative numeric
time-to-event/censoring duration, present under the same name on every node.
`outcome_col` (default OUTCOME_COL) must resolve to a binary 0/1 event
indicator (1 = event observed, 0 = censored) -- either because it already is
one, or because `outcome_representation` (default OUTCOME_REPRESENTATION,
None) says how to derive 0/1 from it, e.g. `{"mode": "presence"}` when the
outcome column's value being present/missing (not its content) indicates the
event, or `{"mode": "equals", "positive_value": ...}` / `{"mode": "isin",
"positive_values": [...]}` when a specific value (or set of values) means the
event occurred. See _derive_binary for the full set of modes. `expl_vars`
(default EXPL_VARS) must be numeric covariates, present under the same names
on every node. A missing column, an unknown representation mode, or a column
that still isn't binary after representation raises a clear ValueError.
Nodes with <= `minimum_events` (default 10, a per-node privacy threshold)
observed events are excluded from the analysis entirely rather than
contributing noisy statistics; `central()` forwards `minimum_events` to
`get_unique_event_times` so it's overridable per run, not just per image
build. `min_organizations` (default 1, i.e. no restriction) is checked
*after* that exclusion: if fewer than `min_organizations` organizations
still qualify once the ones below `minimum_events` are dropped, `central()`
returns a clear error naming the excluded organizations, instead of
silently proceeding on whichever ones happen to qualify -- having enough
organizations nominally included in the task is not the same as having
enough organizations with enough events.

`central()` also takes an `organizations` kwarg (default None): a list of
organization names to restrict the sub-task fan-out to, instead of every
organization in the collaboration. Without it, an organization that's
registered in the collaboration but has no node actually running gets a
sub-task it can never pick up, and client.wait_for_results() then blocks
forever waiting on it -- `organizations` lets the caller exclude it. An
unknown organization name raises a clear ValueError. This is a separate
mechanism from the `minimum_events` exclusion above: `organizations` narrows
the fan-out before any task is even sent, `minimum_events` excludes a node
after its own event-count report comes back.

Covariates are standardized before Newton-Raphson: `central()` derives a
pooled centre (mean) and scale (std-dev) per covariate from the event-case
sums `compute_summed_z` returns (`sum`/`sum_squares`/`count`), and every
`perform_iteration` call fits on `(x - centre) / scale` instead of raw `x`.
The Cox partial likelihood is invariant to this affine transform, so the
fit itself is unchanged -- this is pure numerics, not a privacy or
data-contract feature. It exists because raw covariate scale risks
`exp(X @ beta)` overflow as beta grows during iteration, and very
different covariate scales (e.g. age vs. a lab value in the thousands)
otherwise produce a badly conditioned Hessian. `central()` back-transforms
beta, the standard errors and the Hessian/covariance to original units
once the loop ends, so reported hazard ratios are per original-unit
increase, not per standardized unit. A covariate with ~zero variance among
event cases (or none at all) gets scale=1 (no rescaling) instead of a
division by zero.

`central()` also takes `risk_set_min_change` (default 1, i.e. no restriction)
and `max_held_event_fraction` (default None, i.e. no restriction). At each
unique event time, `agg1`/`agg2` (the risk-set size and covariate sum,
summed across nodes) are released fresh every Newton-Raphson iteration. If
exactly one individual left the risk set since the last released snapshot,
`(agg2(t) - agg2(t')) / (agg1(t) - agg1(t'))` recovers that individual's
exact covariate vector -- a leak specific to this iterative risk-set
release, not present in a single-shot aggregate. `risk_set_min_change`
guards this: if fewer than that many individuals have left the risk set
since the last released snapshot, the previous snapshot is held (repeated)
instead of releasing a new one, so the subtraction above yields zero
departures, not one. Holding a snapshot means later iterations fit against
a slightly stale risk set at that time point, which can degrade the fit if
it happens often; `max_held_event_fraction` aborts the run with a clear
error if the fraction of events scored against a held snapshot exceeds it,
instead of silently returning a degraded fit. Like `minimum_events`/
`min_organizations` above, these are requester-supplied kwargs (set in
run_study.py), not a node-enforced policy.

Every node-side method (`get_unique_event_times`, `compute_summed_z`,
`perform_iteration`) also verifies it was dispatched by `central()` as a
proper vantage6 sub-task, not called directly -- which would bypass every
guard above (`minimum_events`, the risk-set jump guard) entirely, since
those only run inside `central()`'s orchestration. The check decodes the
container's own token to find its task id, looks that task up, and fails
closed (raises) unless the task has a parent (vantage6 only sets one for
a sub-task spawned by a running container, never for a task a user
started directly). The only exemption is `MockAlgorithmClient`, used by
this repo's own unit tests, which has no real parent-task linkage to check.
"""

import math

import jwt
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import chi2, norm
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from vantage6.algorithm.tools.util import error, info, warn

TIME_COL = "Survival.time"
OUTCOME_COL = "deadstatus.event"
OUTCOME_REPRESENTATION = None
EXPL_VARS = ["age", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage"]
MAX_ITERATIONS = 10
TOLERANCE = 1e-6

RISK_SET_MIN_CHANGE = 1
MAX_HELD_EVENT_FRACTION = None

MINIMUM_EVENTS = 10
MIN_ORGANIZATIONS = 1


def ensure_spawned_by_central(client, method_name: str) -> None:
    """Fail closed unless this call was dispatched by central() as a proper
    vantage6 sub-task -- see the module docstring. MockAlgorithmClient
    (this repo's own unit tests) is the only exemption."""
    if isinstance(client, MockAlgorithmClient):
        return
    try:
        payload = jwt.decode(client._access_token, options={"verify_signature": False})
        task_id = payload["sub"]["task_id"]
    except Exception as exc:
        msg = f"{method_name}: could not verify task provenance (undecodable token); refusing to run."
        error(msg)
        raise RuntimeError(msg) from exc
    try:
        task = client.task.get(task_id)
    except Exception as exc:
        msg = f"{method_name}: could not verify task provenance (task lookup failed); refusing to run."
        error(msg)
        raise RuntimeError(msg) from exc
    if not (task.get("parent") or task.get("parent_id")):
        msg = (
            f"{method_name}: this method must be dispatched by central() as a sub-task, "
            f"not called directly -- doing so bypasses central()'s guards."
        )
        error(msg)
        raise RuntimeError(msg)


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


def _derive_binary(series: pd.Series, representation: dict | None, column_name: str) -> pd.Series:
    """Resolve a raw column to 0/1 (NaN preserved for later dropna), either as-is or
    via a representation spec: {"mode": "presence"} (1 if not null, 0 if null),
    {"mode": "equals", "positive_value": ...} (1 if value matches, else 0), or
    {"mode": "isin", "positive_values": [...]} (1 if value is one of these, else 0).
    Any mode accepts "invert": true to flip the result. Raises ValueError if the
    resulting column (after representation, or as-is when none is given) isn't
    numeric and binary."""
    if not representation:
        result = series
    else:
        mode = representation.get("mode")
        if mode == "presence":
            result = series.notna().astype(float)
        elif mode == "equals":
            positive_value = representation["positive_value"]
            result = series.where(series.isna(), (series == positive_value).astype(float))
        elif mode == "isin":
            positive_values = representation["positive_values"]
            result = series.where(series.isna(), series.isin(positive_values).astype(float))
        else:
            msg = (
                f"column {column_name!r}: unknown representation mode {mode!r}; "
                f"expected one of 'presence', 'equals', 'isin'"
            )
            error(msg)
            raise ValueError(msg)
        if representation.get("invert"):
            result = result.where(result.isna(), 1.0 - result)

    try:
        numeric = pd.to_numeric(result, errors="raise")
    except (ValueError, TypeError) as exc:
        msg = (
            f"column {column_name!r} is not numeric ({exc}); if it isn't already binary "
            f"(0/1), derive it with a representation dict, e.g. {{'mode': 'presence'}} if "
            f"the value's presence/absence indicates the event"
        )
        error(msg)
        raise ValueError(msg) from exc

    non_null = numeric.dropna()
    bad_values = sorted(non_null.loc[~non_null.isin([0.0, 1.0])].unique().tolist())
    if bad_values:
        msg = (
            f"column {column_name!r} contains non-binary value(s) {bad_values[:5]} "
            f"(representation={representation!r}); derive it to 0/1 first via a "
            f"representation dict, e.g. {{'mode': 'equals', 'positive_value': ...}} or "
            f"{{'mode': 'isin', 'positive_values': [...]}}"
        )
        error(msg)
        raise ValueError(msg)
    return numeric


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        warn("Matrix inversion failed; using pseudo-inverse")
        return np.linalg.pinv(matrix)


def apply_risk_set_jump_guard(
    agg1: np.ndarray,
    agg2: np.ndarray,
    agg3: np.ndarray,
    at_risk_counts: np.ndarray,
    min_change: int,
) -> tuple:
    """Hold the previous released (agg1, agg2, agg3) snapshot at any time
    index where fewer than `min_change` individuals have left the risk set
    since the last released snapshot, instead of releasing the fresh one --
    see the module docstring for why. `at_risk_counts` is the plain (not
    exp-weighted) risk-set size at each time index, aggregated the same way
    as `agg1`; it's what "how many individuals left" is computed from.
    Applied once centrally on the already-summed-across-nodes arrays: the
    leak targets the released aggregate, so guarding the aggregate directly
    gives the same protection as guarding each node's own contribution
    before summing, with one guard instead of coordinating N independent
    per-node ones.

    Returns (agg1, agg2, agg3, held_mask) where held_mask[i] is True if
    index i's snapshot was held rather than freshly released.
    """
    n_times = len(at_risk_counts)
    held_mask = np.zeros(n_times, dtype=bool)
    if min_change <= 1 or n_times == 0:
        return agg1, agg2, agg3, held_mask

    agg1 = agg1.copy()
    agg2 = agg2.copy()
    agg3 = agg3.copy()
    last_released_count = at_risk_counts[0]
    for i in range(1, n_times):
        departed = last_released_count - at_risk_counts[i]
        if departed < min_change:
            agg1[i] = agg1[i - 1]
            agg2[i] = agg2[i - 1]
            agg3[i] = agg3[i - 1]
            held_mask[i] = True
        else:
            last_released_count = at_risk_counts[i]
    return agg1, agg2, agg3, held_mask


def compute_derivatives(
    summed_agg1: np.ndarray,
    summed_agg2: np.ndarray,
    summed_agg3: np.ndarray,
    aggregated_time_events: pd.DataFrame,
    z_sum: pd.Series,
) -> tuple:
    z_sum_vec = z_sum.to_numpy(dtype=float)
    n_covariates = z_sum_vec.shape[0]
    tot_p1 = np.zeros(n_covariates, dtype=float)
    tot_p2 = np.zeros((n_covariates, n_covariates), dtype=float)

    for index, row in aggregated_time_events.iterrows():
        denom = float(summed_agg1[index])
        if denom <= 0.0:
            continue
        freq = float(row["freq"])
        s1 = freq * (summed_agg2[index] / denom)
        first_part = summed_agg3[index] / denom
        numerator = np.outer(summed_agg2[index], summed_agg2[index])
        second_part = numerator / (denom * denom)
        s2 = freq * (first_part - second_part)
        tot_p1 += s1
        tot_p2 += s2

    primary_derivative = z_sum_vec - tot_p1
    secondary_derivative = -tot_p2
    return primary_derivative, secondary_derivative


def _compute_log_likelihood(
    z_sum: pd.Series,
    beta: np.ndarray,
    summed_agg1: np.ndarray,
    aggregated_time_events: pd.DataFrame,
) -> float:
    linear_part = float(np.dot(z_sum.to_numpy(dtype=float), beta))
    risk_set_part = 0.0
    for i, row in aggregated_time_events.iterrows():
        if i >= len(summed_agg1):
            break
        denom = float(summed_agg1[i])
        if denom <= 0.0:
            warn(f"Risk set denominator non-positive at index {i}: {denom}")
            continue
        risk_set_part += float(row["freq"]) * math.log(denom)
    return linear_part - risk_set_part


def _build_results_table(beta: np.ndarray, s_errors: np.ndarray, expl_vars: list) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        zvalues = np.divide(
            beta,
            s_errors,
            out=np.zeros_like(beta, dtype=float),
            where=s_errors > 0,
        )
    pvalues = 2.0 * norm.sf(np.abs(zvalues))

    results = pd.DataFrame(
        {
            "Coef": np.around(beta, 5),
            "Exp(coef)": np.around(np.exp(beta), 5),
            "SE": np.around(s_errors, 5),
            "Var": expl_vars,
            "Z": zvalues,
            "p-value": pvalues,
        }
    )
    results["lower_CI"] = np.around(np.exp(results["Coef"] - 1.96 * results["SE"]), 5)
    results["upper_CI"] = np.around(np.exp(results["Coef"] + 1.96 * results["SE"]), 5)
    return results.set_index("Var")


@algorithm_client
def central(
    client: AlgorithmClient,
    time_col: str = TIME_COL,
    outcome_col: str = OUTCOME_COL,
    outcome_representation: dict = OUTCOME_REPRESENTATION,
    expl_vars: list = None,
    organizations: list | None = None,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
    risk_set_min_change: int = RISK_SET_MIN_CHANGE,
    max_held_event_fraction: float | None = MAX_HELD_EVENT_FRACTION,
    minimum_events: int = MINIMUM_EVENTS,
    min_organizations: int = MIN_ORGANIZATIONS,
) -> dict:
    expl_vars = list(expl_vars) if expl_vars else list(EXPL_VARS)
    n_covs = len(expl_vars)
    max_iterations = max(1, int(max_iterations))
    tolerance = float(tolerance)
    risk_set_min_change = int(risk_set_min_change)
    if risk_set_min_change < 1:
        return {"error": f"'risk_set_min_change' must be at least 1, got {risk_set_min_change}"}
    if max_held_event_fraction is not None and not (0.0 <= max_held_event_fraction <= 1.0):
        return {
            "error": f"'max_held_event_fraction' must be between 0 and 1, got {max_held_event_fraction}"
        }
    min_organizations = int(min_organizations)
    if min_organizations < 1:
        return {"error": f"'min_organizations' must be at least 1, got {min_organizations}"}

    orgs = _filter_organizations(client.organization.list(), organizations, "central")
    ids = [org["id"] for org in orgs]
    org_name_by_id = {org["id"]: org["name"] for org in orgs}

    if len(ids) < min_organizations:
        return {
            "error": (
                f"Only {len(ids)} organization(s) are included in this task, but "
                f"'min_organizations' requires at least {min_organizations}."
            )
        }

    info("=" * 60)
    info("FEDERATED COX PROPORTIONAL HAZARDS REGRESSION")
    info("=" * 60)
    info(f"Organizations : {[org['name'] for org in orgs]}")
    info(f"Time column   : {time_col}")
    info(f"Outcome column: {outcome_col}  (representation={outcome_representation})")
    info(f"Covariates    : {expl_vars}")

    # ── Phase 1: unique event times ────────────────────────────────────────
    info("")
    info("── PHASE 1: Determine unique event times ────────────────────")
    times_task = client.task.create(
        input_={
            "method": "get_unique_event_times",
            "kwargs": {
                "time_col": time_col,
                "outcome_col": outcome_col,
                "outcome_representation": outcome_representation,
                "minimum_events": minimum_events,
            },
        },
        organizations=ids,
        name="coxph_unique_event_times",
        description="Get unique event times and frequencies",
    )
    times_results = client.wait_for_results(times_task["id"])

    excluded_ids = []
    unique_frames = []
    for output in times_results:
        not_met = output.get("n_threshold_not_met")
        if not_met is not None:
            warn(f"Insufficient events for organization {not_met}; excluding from analysis.")
            excluded_ids.append(not_met)
            continue
        times = output.get("times")
        if times:
            unique_frames.append(pd.DataFrame.from_dict(times))

    ids = [org_id for org_id in ids if org_id not in excluded_ids]
    if len(ids) < min_organizations:
        excluded_names = [org_name_by_id.get(org_id, org_id) for org_id in excluded_ids]
        return {
            "error": (
                f"Only {len(ids)} organization(s) have at least 'minimum_events'="
                f"{minimum_events} events, but 'min_organizations' requires at least "
                f"{min_organizations}. Organization(s) below the event threshold: "
                f"{excluded_names}."
            )
        }
    if not unique_frames:
        return {"error": "No usable event-time data across any organization"}

    aggregated_time_events = pd.concat(unique_frames).groupby(time_col, as_index=False).sum()
    unique_time_events = aggregated_time_events[time_col].tolist()
    info(f"Included organizations: {ids} (excluded: {excluded_ids})")
    info(f"Unique event times    : {len(unique_time_events)}")

    # ── Phase 2: summed Z statistic ─────────────────────────────────────────
    info("")
    info("── PHASE 2: Compute summed Z statistic ──────────────────────")
    z_task = client.task.create(
        input_={
            "method": "compute_summed_z",
            "kwargs": {
                "outcome_col": outcome_col,
                "outcome_representation": outcome_representation,
                "expl_vars": expl_vars,
            },
        },
        organizations=ids,
        name="coxph_summed_z",
        description="Compute summed z statistic",
    )
    z_results = client.wait_for_results(z_task["id"])

    z_sum = pd.Series(0.0, index=expl_vars)
    z_sum_squares = pd.Series(0.0, index=expl_vars)
    z_count = pd.Series(0.0, index=expl_vars)
    for output in z_results:
        z_sum += pd.Series(output["sum"], index=expl_vars, dtype=float).fillna(0.0)
        z_sum_squares += pd.Series(output["sum_squares"], index=expl_vars, dtype=float).fillna(0.0)
        z_count += pd.Series(output["count"], index=expl_vars, dtype=float).fillna(0.0)

    # Pooled per-covariate mean/variance over event-case rows -- see the
    # module docstring. A covariate with no non-null event-case value (or
    # zero variance among them) keeps centre=0/scale=1, i.e. unstandardized.
    safe_count = z_count.where(z_count > 0, np.nan)
    centre = (z_sum / safe_count).fillna(0.0)
    variance = (z_sum_squares / safe_count - centre**2).fillna(0.0).clip(lower=0.0)
    scale = np.sqrt(variance).where(lambda s: s > 1e-8, 1.0)
    centre_vec = centre.reindex(expl_vars).to_numpy(dtype=float)
    scale_vec = scale.reindex(expl_vars).to_numpy(dtype=float)
    z_sum_std = (z_sum - z_count * centre) / scale

    # ── Phase 3: Newton-Raphson iterations ───────────────────────────────────
    info("")
    info("── PHASE 3: Newton-Raphson iterations ───────────────────────")
    beta = np.zeros(n_covs, dtype=float)
    secondary_derivative = -np.eye(n_covs, dtype=float)
    summed_agg1 = np.zeros(len(unique_time_events), dtype=float)

    for iteration in range(max_iterations):
        iteration_task = client.task.create(
            input_={
                "method": "perform_iteration",
                "kwargs": {
                    "time_col": time_col,
                    "expl_vars": expl_vars,
                    "beta": beta.tolist(),
                    "unique_time_events": unique_time_events,
                    "centre": centre_vec.tolist(),
                    "scale": scale_vec.tolist(),
                },
            },
            organizations=ids,
            name=f"coxph_iteration_{iteration + 1}",
            description="Iterating to find the optimal beta",
        )
        results = client.wait_for_results(iteration_task["id"])

        n_times = len(unique_time_events)
        summed_agg1 = np.zeros(n_times, dtype=float)
        summed_agg2 = np.zeros((n_times, n_covs), dtype=float)
        summed_agg3 = np.zeros((n_times, n_covs, n_covs), dtype=float)
        summed_at_risk_counts = np.zeros(n_times, dtype=float)

        for output in results:
            summed_agg1 += np.asarray(output["agg1"], dtype=float)
            agg2_df = pd.DataFrame.from_dict(output["agg2"]).reindex(columns=expl_vars)
            summed_agg2 += agg2_df.to_numpy(dtype=float)
            summed_agg3 += np.asarray(output["agg3"], dtype=float)
            summed_at_risk_counts += np.asarray(output["at_risk_counts"], dtype=float)

        summed_agg1, summed_agg2, summed_agg3, held_mask = apply_risk_set_jump_guard(
            summed_agg1, summed_agg2, summed_agg3, summed_at_risk_counts, risk_set_min_change
        )
        if max_held_event_fraction is not None and held_mask.any():
            held_fraction = float(aggregated_time_events.loc[held_mask, "freq"].sum()) / float(
                aggregated_time_events["freq"].sum()
            )
            if held_fraction > max_held_event_fraction:
                return {
                    "error": (
                        f"{held_fraction:.1%} of events were scored against a risk-set "
                        f"snapshot held by the 'risk_set_min_change' guard, exceeding "
                        f"'max_held_event_fraction'={max_held_event_fraction:.1%}. Lower "
                        f"'risk_set_min_change' or raise 'max_held_event_fraction' to proceed."
                    )
                }

        primary_derivative, secondary_derivative = compute_derivatives(
            summed_agg1=summed_agg1,
            summed_agg2=summed_agg2,
            summed_agg3=summed_agg3,
            aggregated_time_events=aggregated_time_events,
            z_sum=z_sum_std,
        )

        beta_old = beta.copy()
        try:
            beta = beta_old - solve(secondary_derivative, primary_derivative)
        except np.linalg.LinAlgError:
            warn("Hessian is singular; falling back to pseudo-inverse update")
            beta = beta_old - _safe_inverse(secondary_derivative).dot(primary_derivative)

        delta = float(np.max(np.abs(beta - beta_old)))
        info(f"Iteration {iteration + 1}: max |Δbeta| = {delta:.3e}")
        if math.isnan(delta):
            warn("Optimization update produced NaN delta; stopping iterations")
            break
        if delta <= tolerance:
            info("Betas have settled; optimization converged")
            break

    # ── Phase 4: inference ───────────────────────────────────────────────────
    # The log-likelihood is invariant to the standardizing affine transform
    # (see module docstring), so it must be computed from the standardized
    # beta/z_sum/summed_agg1 that are actually consistent with each other --
    # not from the back-transformed beta below, which has no matching
    # standardized-space summed_agg1 of its own.
    try:
        log_likelihood = _compute_log_likelihood(
            z_sum=z_sum_std,
            beta=beta,
            summed_agg1=summed_agg1,
            aggregated_time_events=aggregated_time_events,
        )
        if math.isnan(log_likelihood) or math.isinf(log_likelihood):
            raise ValueError(f"Invalid log-likelihood: {log_likelihood}")
        aic = float(-2.0 * log_likelihood + 2.0 * len(beta))
    except (ValueError, FloatingPointError) as exc:
        warn(f"Could not compute AIC due to numerical/data issue: {exc}")
        aic = float("nan")

    # Back-transform beta/Hessian/covariance from standardized to original
    # covariate units: beta_x = beta_std / scale, information_x = information_std
    # * outer(scale, scale), covariance_x = covariance_std / outer(scale, scale)
    # -- see the module docstring for the derivation.
    scale_outer = np.outer(scale_vec, scale_vec)
    information_std = -secondary_derivative
    fisher_std = _safe_inverse(information_std)
    beta = beta / scale_vec
    fisher = fisher_std / scale_outer
    information = information_std * scale_outer
    s_errors = np.sqrt(np.clip(np.diag(fisher), a_min=0.0, a_max=None))

    wald_statistic = float(beta @ information @ beta)
    overall_p_value = float(chi2.sf(wald_statistic, len(beta)))

    results = _build_results_table(beta=beta, s_errors=s_errors, expl_vars=expl_vars)

    model_warnings = []
    for covariate, row in results.iterrows():
        coef = float(row["Coef"])
        se = float(row["SE"])
        if abs(coef) > 10.0 or np.isinf(coef) or np.isnan(coef) or abs(se) > 10.0 or np.isinf(se) or np.isnan(se):
            msg = (
                f"Warning: Covariate '{covariate}' may perfectly predict the event "
                f"(coef={coef}, SE={se}). Results may be unreliable."
            )
            warn(msg)
            model_warnings.append(msg)

    info(f"Final coefficients: {beta.tolist()}")
    return {
        "included_organizations": ids,
        "excluded_organizations": excluded_ids,
        "model": results.to_json(),
        "overall_p_value": overall_p_value,
        "aic": aic,
        "degrees_of_freedom": len(beta),
        "warnings": model_warnings,
    }


@data(1)
@algorithm_client
def get_unique_event_times(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    outcome_col: str = OUTCOME_COL,
    outcome_representation: dict = OUTCOME_REPRESENTATION,
    minimum_events: int = MINIMUM_EVENTS,
) -> dict:
    ensure_spawned_by_central(client, "get_unique_event_times")
    _require_columns(df, [time_col, outcome_col], "get_unique_event_times")
    df = df.assign(**{outcome_col: _derive_binary(df[outcome_col], outcome_representation, outcome_col)})
    info("Computing unique event times")
    if int(df[outcome_col].notnull().sum()) <= int(minimum_events):
        org_id = getattr(client, "organization_id", -1)
        warn("Sub-task skipped because the number of samples is too small")
        return {"n_threshold_not_met": int(org_id)}

    times = df[df[outcome_col] == 1].groupby(time_col, as_index=False).count()
    times = times.sort_values(by=time_col)[[time_col, outcome_col]]
    times["freq"] = times[outcome_col]
    times = times.drop(columns=outcome_col)
    info(f"Local unique event times: {len(times)}")
    return {"times": times.to_dict()}


@data(1)
@algorithm_client
def compute_summed_z(
    client: AlgorithmClient,
    df: pd.DataFrame,
    outcome_col: str = OUTCOME_COL,
    outcome_representation: dict = OUTCOME_REPRESENTATION,
    expl_vars: list = None,
) -> dict:
    ensure_spawned_by_central(client, "compute_summed_z")
    expl_vars = list(expl_vars) if expl_vars else list(EXPL_VARS)
    _require_columns(df, [outcome_col, *expl_vars], "compute_summed_z")
    df = df.assign(**{outcome_col: _derive_binary(df[outcome_col], outcome_representation, outcome_col)})
    info("Computing summed z statistics")
    # sum_squares/count (per covariate, NaNs skipped independently per column
    # like sum already is) let central() derive a pooled mean/variance per
    # covariate for standardization, without assuming every covariate has the
    # same missingness pattern among event rows.
    event_rows = df[df[outcome_col] == 1][expl_vars].astype(float)
    z_sum = event_rows.sum(skipna=True).to_dict()
    z_sum_squares = (event_rows**2).sum(skipna=True).to_dict()
    z_count = event_rows.notna().sum().astype(float).to_dict()
    return {"sum": z_sum, "sum_squares": z_sum_squares, "count": z_count}


@data(1)
@algorithm_client
def perform_iteration(
    client: AlgorithmClient,
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    expl_vars: list = None,
    beta: list = None,
    unique_time_events: list = None,
    centre: list = None,
    scale: list = None,
) -> dict:
    ensure_spawned_by_central(client, "perform_iteration")
    expl_vars = list(expl_vars) if expl_vars else list(EXPL_VARS)
    _require_columns(df, [time_col, *expl_vars], "perform_iteration")
    if (centre is None) != (scale is None):
        raise ValueError("'centre' and 'scale' must both be given, or both omitted")
    beta_arr = np.asarray(beta, dtype=float)
    info("Computing aggregates for the derivation of the partial likelihood")

    working = df[[time_col, *expl_vars]].dropna(how="any")
    X = working[expl_vars].to_numpy(dtype=float)
    if centre is not None:
        # Standardized space: beta (and therefore the aggregates below) are
        # in standardized units until central() back-transforms the final
        # result -- see the module docstring.
        X = (X - np.asarray(centre, dtype=float)) / np.asarray(scale, dtype=float)
    times = working[time_col].to_numpy(dtype=float)

    if X.shape[0] == 0:
        zeros = np.zeros((len(unique_time_events), len(expl_vars)))
        return {
            "agg1": [0.0] * len(unique_time_events),
            "agg2": pd.DataFrame(zeros, columns=expl_vars).to_dict(),
            "agg3": [np.zeros((len(expl_vars), len(expl_vars))).tolist() for _ in unique_time_events],
            "at_risk_counts": [0] * len(unique_time_events),
        }

    exp_xb = np.exp(X @ beta_arr)
    agg1 = []
    agg2_rows = []
    agg3 = []
    at_risk_counts = []
    n_covariates = len(expl_vars)

    for unique_time in unique_time_events:
        mask = times >= float(unique_time)
        at_risk_counts.append(int(mask.sum()))
        if not np.any(mask):
            agg1.append(0.0)
            agg2_rows.append(np.zeros(n_covariates, dtype=float))
            agg3.append(np.zeros((n_covariates, n_covariates), dtype=float))
            continue

        Xi = X[mask]
        exp_i = exp_xb[mask]
        weighted = Xi * exp_i[:, None]

        agg1.append(float(exp_i.sum()))
        agg2_rows.append(weighted.sum(axis=0))
        agg3.append(Xi.T @ weighted)

    agg2_df = pd.DataFrame(agg2_rows, columns=expl_vars)
    return {
        "agg1": agg1,
        "agg2": agg2_df.to_dict(),
        "agg3": [matrix.tolist() for matrix in agg3],
        # Plain (not exp-weighted) risk-set size at each time -- used only by
        # central()'s risk-set jump guard to decide how many individuals left
        # the risk set between releases; see the module docstring.
        "at_risk_counts": at_risk_counts,
    }
