"""Federated average of a single column.

Data contract: `column` (central()/partial() kwarg, default AVERAGE_VAR) must
be a numeric column, present under the same name on every node. No encoding
or cleaning is done here -- a missing column raises a clear ValueError; a
present-but-non-numeric column will fail less clearly, inside the sum/count
computation itself.

`central()` also takes an `organizations` kwarg (default None): a list of
organization names to restrict the sub-task fan-out to, instead of every
organization in the collaboration. Without it, an organization that's
registered in the collaboration but has no node actually running gets a
sub-task it can never pick up, and client.wait_for_results() then blocks
forever waiting on it -- `organizations` lets the caller exclude it. An
unknown organization name raises a clear ValueError.

`central()` also takes `min_organizations` (default 1, i.e. no restriction)
and `min_rows` (default 0, i.e. no restriction). Any organization reporting
fewer than `min_rows` usable (non-null) values for `column` is excluded
entirely from the average -- not just hidden from `n`. If fewer than
`min_organizations` organizations remain after that exclusion, `central()`
raises a clear `ValueError` naming the organizations that fell short,
instead of silently averaging over whichever ones happen to qualify --
enough organizations nominally included in the task is not the same as
enough organizations with enough data. This composes with `organizations`
above rather than replacing it: `organizations` narrows the fan-out before
any task is even sent, `min_rows`/`min_organizations` exclude a node after
its own row count comes back.
"""

import pandas as pd
from vantage6.algorithm.tools.util import info, error
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.client import AlgorithmClient

AVERAGE_VAR = "age"
MIN_ORGANIZATIONS = 1
MIN_ROWS = 0


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


@algorithm_client
def central(
    client: AlgorithmClient,
    column: str = AVERAGE_VAR,
    organizations: list | None = None,
    min_organizations: int = MIN_ORGANIZATIONS,
    min_rows: int = MIN_ROWS,
) -> dict:
    min_organizations = int(min_organizations)
    if min_organizations < 1:
        msg = f"central: 'min_organizations' must be at least 1, got {min_organizations}"
        error(msg)
        raise ValueError(msg)
    min_rows = int(min_rows)
    if min_rows < 0:
        msg = f"central: 'min_rows' must be >= 0, got {min_rows}"
        error(msg)
        raise ValueError(msg)

    orgs = _filter_organizations(client.organization.list(), organizations, "central")
    org_ids = [org["id"] for org in orgs]
    org_name_by_id = {org["id"]: org["name"] for org in orgs}

    if len(org_ids) < min_organizations:
        msg = (
            f"central: only {len(org_ids)} organization(s) are included in this task, but "
            f"'min_organizations' requires at least {min_organizations}."
        )
        error(msg)
        raise ValueError(msg)

    info(f"Submitting partial tasks to {len(org_ids)} organizations: {org_ids}")

    task = client.task.create(
        input_={"method": "partial", "kwargs": {"column": column}},
        organizations=org_ids,
        name="average_partial",
        description=f"Compute local {column} sum and count",
    )
    info(f"Waiting for partial results (task id={task['id']})")
    partials = client.wait_for_results(task["id"])

    # Organizations below min_rows are dropped entirely -- their sum/count
    # must not contribute to the average at all, not just be excluded from n.
    qualifying = [r for r in partials if r["count"] >= min_rows]
    qualifying_ids = [r["organization_id"] for r in qualifying]

    if len(qualifying_ids) < min_organizations:
        below_threshold = [
            org_name_by_id.get(r["organization_id"], r["organization_id"])
            for r in partials
            if r["count"] < min_rows
        ]
        msg = (
            f"central: only {len(qualifying_ids)} organization(s) have at least 'min_rows'="
            f"{min_rows} usable rows, but 'min_organizations' requires at least "
            f"{min_organizations}. Organization(s) below the row threshold: {below_threshold}."
        )
        error(msg)
        raise ValueError(msg)

    total_sum = sum(r["sum"] for r in qualifying)
    total_count = sum(r["count"] for r in qualifying)

    if total_count == 0:
        return {"average": None, "variable": column, "n": 0}

    average = total_sum / total_count
    info(f"Global average {column} = {average:.4f} (n={total_count})")
    return {"average": average, "variable": column, "n": total_count}


@data(1)
@algorithm_client
def partial(client: AlgorithmClient, df: pd.DataFrame, column: str = AVERAGE_VAR) -> dict:
    _require_columns(df, [column], "partial")
    info(f"Processing {len(df)} rows for column '{column}'")
    val_sum = float(df[column].sum())
    val_count = int(df[column].count())
    info(f"Local sum={val_sum:.4f}, count={val_count}")
    return {"sum": val_sum, "count": val_count, "organization_id": client.organization_id}
