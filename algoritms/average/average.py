import pandas as pd
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.client import AlgorithmClient

AVERAGE_VAR = "age"

@algorithm_client
def central(client: AlgorithmClient, column: str = AVERAGE_VAR) -> dict:
    orgs = client.organization.list()
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
    info(f"Processing {len(df)} rows for column '{column}'")
    val_sum = float(df[column].sum())
    val_count = int(df[column].count())
    info(f"Local sum={val_sum:.4f}, count={val_count}")
    return {"sum": val_sum, "count": val_count}
