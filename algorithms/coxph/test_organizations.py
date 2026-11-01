import pandas as pd
import pytest
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from coxph import _filter_organizations


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="coxph",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _survival_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": list(range(1, n + 1)),
            "event": [1, 0] * (n // 2),
            "age": list(range(40, 40 + n)),
        }
    )


def test_filter_organizations_none_returns_all() -> None:
    orgs = [{"id": 0, "name": "alpha"}, {"id": 1, "name": "beta"}]
    assert _filter_organizations(orgs, None, "central") == orgs


def test_filter_organizations_restricts_to_named_orgs() -> None:
    orgs = [{"id": 0, "name": "alpha"}, {"id": 1, "name": "beta"}]
    assert _filter_organizations(orgs, ["beta"], "central") == [{"id": 1, "name": "beta"}]


def test_filter_organizations_raises_for_unknown_name() -> None:
    with pytest.raises(ValueError, match=r"ghost"):
        _filter_organizations([{"id": 0, "name": "alpha"}], ["ghost"], "central")


def test_filter_organizations_raises_for_empty_list() -> None:
    with pytest.raises(ValueError, match=r"empty list"):
        _filter_organizations([{"id": 0, "name": "alpha"}], [], "central")


def test_central_organizations_raises_clear_error_for_unknown_org() -> None:
    df = _survival_df(30)
    with pytest.raises(ValueError, match=r"nonexistent-org"):
        _run_central(
            [df], time_col="time", outcome_col="event", expl_vars=["age"],
            organizations=["nonexistent-org"],
        )


def test_central_organizations_excludes_the_other_node() -> None:
    included = _survival_df(30)
    excluded = _survival_df(30)  # would double the counts if wrongly included too
    result = _run_central(
        [included, excluded], time_col="time", outcome_col="event", expl_vars=["age"],
        organizations=["mock-0"],
    )
    assert result["included_organizations"] == [0]
