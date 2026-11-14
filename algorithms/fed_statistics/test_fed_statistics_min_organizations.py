import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="fed_statistics",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _make_df(n_rows: int, start: float) -> pd.DataFrame:
    return pd.DataFrame({"age": [start + i for i in range(n_rows)]})


def test_min_organizations_validation_rejects_less_than_one() -> None:
    df = _make_df(20, 20.0)
    result = _run_central([df], statistics={"age": ["mean"]}, min_organizations=0)
    assert "min_organizations" in result["error"]


def test_min_rows_validation_rejects_negative() -> None:
    df = _make_df(20, 20.0)
    result = _run_central([df], statistics={"age": ["mean"]}, min_rows=-1)
    assert "min_rows" in result["error"]


def test_min_organizations_fails_fast_when_task_has_too_few_orgs() -> None:
    org0 = _make_df(20, 20.0)
    org1 = _make_df(20, 100.0)
    result = _run_central(
        [org0, org1], statistics={"age": ["mean"]}, organizations=["mock-0"], min_organizations=2
    )
    assert "error" in result
    assert "min_organizations" in result["error"]


def test_min_rows_and_min_organizations_together_requires_qualifying_orgs() -> None:
    # 3 organizations are nominally included -- enough to satisfy
    # min_organizations=2 on its own -- but only one has enough rows to
    # clear min_rows=10. Having enough organizations included is not the
    # same as having enough organizations with enough rows.
    org0 = _make_df(20, 20.0)  # qualifies
    org1 = _make_df(3, 1000.0)  # excluded
    org2 = _make_df(3, 2000.0)  # excluded

    result = _run_central(
        [org0, org1, org2],
        statistics={"age": ["mean"]},
        options={"suppress_threshold": 0},
        min_rows=10,
        min_organizations=2,
    )
    assert "error" in result
    assert "min_organizations" in result["error"]
    assert "min_rows" in result["error"]
    assert "mock-1" in result["error"]
    assert "mock-2" in result["error"]


def test_min_rows_excludes_low_row_organizations_from_every_column_and_statistic() -> None:
    org0 = _make_df(20, 20.0)  # ages 20..39, qualifies
    org1 = pd.DataFrame({"age": [9999.0, 9999.0, 9999.0]})  # excluded, huge outlier

    result = _run_central(
        [org0, org1],
        statistics={"age": ["mean", "minmax", "nrows"]},
        options={"suppress_threshold": 0},
        min_rows=10,
        min_organizations=1,
    )
    assert "error" not in result
    columns = result["columns"]["age"]
    assert columns["nrows"] == 20
    # The excluded organization's huge outlier values must not leak into any
    # statistic, not just be hidden from the row count.
    assert columns["minmax"]["max"] < 100
    assert columns["mean"] < 100
