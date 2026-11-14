import pandas as pd
import pytest
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="average",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _make_df(n_rows: int, start: float) -> pd.DataFrame:
    return pd.DataFrame({"age": [start + i for i in range(n_rows)]})


def test_min_organizations_validation_rejects_less_than_one() -> None:
    with pytest.raises(ValueError, match="min_organizations"):
        _run_central([_make_df(20, 20.0)], column="age", min_organizations=0)


def test_min_rows_validation_rejects_negative() -> None:
    with pytest.raises(ValueError, match="min_rows"):
        _run_central([_make_df(20, 20.0)], column="age", min_rows=-1)


def test_min_organizations_fails_fast_when_task_has_too_few_orgs() -> None:
    org0 = _make_df(20, 20.0)
    org1 = _make_df(20, 100.0)
    with pytest.raises(ValueError, match="min_organizations"):
        _run_central(
            [org0, org1], column="age", organizations=["mock-0"], min_organizations=2
        )


def test_min_rows_and_min_organizations_together_requires_qualifying_orgs() -> None:
    # 3 organizations are nominally included -- enough to satisfy
    # min_organizations=2 on its own -- but only one has enough rows to
    # clear min_rows=10. Having enough organizations included is not the
    # same as having enough organizations with enough rows.
    org0 = _make_df(20, 20.0)  # qualifies
    org1 = _make_df(3, 1000.0)  # excluded
    org2 = _make_df(3, 2000.0)  # excluded

    with pytest.raises(ValueError, match="min_organizations") as exc_info:
        _run_central([org0, org1, org2], column="age", min_rows=10, min_organizations=2)
    message = str(exc_info.value)
    assert "min_rows" in message
    assert "mock-1" in message
    assert "mock-2" in message


def test_min_rows_excludes_low_row_organizations_from_the_average() -> None:
    org0 = _make_df(20, 20.0)  # ages 20..39, qualifies
    org1 = pd.DataFrame({"age": [9999.0, 9999.0, 9999.0]})  # excluded, huge outlier

    result = _run_central([org0, org1], column="age", min_rows=10, min_organizations=1)
    assert result["n"] == 20
    assert result["average"] < 100
