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
"""

import pandas as pd
from vantage6.algorithm.tools.util import info, error
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.client import AlgorithmClient

AVERAGE_VAR = "age"


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
def central(client: AlgorithmClient, column: str = AVERAGE_VAR, organizations: list | None = None) -> dict:
    orgs = _filter_organizations(client.organization.list(), organizations, "central")
    org_ids = [org["id"] for org in orgs]
    info(f"Submitting partial tasks to {len(org_ids)} organizations: {org_ids}")

    task = client.task.create(
        input_={"method": "partial", "kwargs": {"column": column}},
        organizations=org_ids,
        name="average_partial",
        description=f"Compute local {column} sum and count",
    )
    info(f"Waiting for partial results (task id={task['id']})")
    partials = client.wait_for_results(task["id"])

    total_sum = sum(r["sum"] for r in partials)
    total_count = sum(r["count"] for r in partials)

    if total_count == 0:
        return {"average": None, "variable": column, "n": 0}

    average = total_sum / total_count
    info(f"Global average {column} = {average:.4f} (n={total_count})")
    return {"average": average, "variable": column, "n": total_count}


@data(1)
def partial(df: pd.DataFrame, column: str = AVERAGE_VAR) -> dict:
    _require_columns(df, [column], "partial")
    info(f"Processing {len(df)} rows for column '{column}'")
    val_sum = float(df[column].sum())
    val_count = int(df[column].count())
    info(f"Local sum={val_sum:.4f}, count={val_count}")
    return {"sum": val_sum, "count": val_count}
