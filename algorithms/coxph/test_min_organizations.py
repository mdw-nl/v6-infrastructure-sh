import numpy as np
import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="coxph",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _survival_df(n_events: int, seed: int) -> pd.DataFrame:
    # n_events rows, every row an observed event, so "events" == "rows" here
    # and minimum_events thresholds this dataset's row count directly.
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time": rng.integers(1, 200, size=n_events).astype(float),
            "event": [1] * n_events,
            "age": 40 + rng.integers(0, 50, size=n_events),
        }
    )


def test_minimum_events_is_forwarded_and_overridable() -> None:
    # 8 events would fail the module default (minimum_events=10) but must
    # pass once a run explicitly lowers it -- proving it's actually forwarded
    # from central(), not just a hardcoded partial default.
    df = _survival_df(8, seed=1)
    result = _run_central(
        [df], time_col="time", outcome_col="event", expl_vars=["age"], minimum_events=2
    )
    assert "error" not in result
    assert result["included_organizations"] == [0]


def test_min_organizations_validation_rejects_less_than_one() -> None:
    df = _survival_df(20, seed=1)
    result = _run_central(
        [df], time_col="time", outcome_col="event", expl_vars=["age"], min_organizations=0
    )
    assert "min_organizations" in result["error"]


def test_min_organizations_fails_fast_when_task_has_too_few_orgs() -> None:
    org0 = _survival_df(20, seed=1)
    org1 = _survival_df(20, seed=2)
    result = _run_central(
        [org0, org1], time_col="time", outcome_col="event", expl_vars=["age"],
        organizations=["mock-0"], min_organizations=2,
    )
    assert "error" in result
    assert "min_organizations" in result["error"]


def test_minimum_events_and_min_organizations_together_requires_qualifying_orgs() -> None:
    # The scenario min_organizations alone can't catch: 3 organizations are
    # nominally included -- enough to satisfy min_organizations=2 on its
    # own -- but only one of them has enough *events* to clear
    # minimum_events=10. Having enough organizations included is not the
    # same as having enough organizations with enough events.
    org0 = _survival_df(20, seed=1)  # qualifies
    org1 = _survival_df(3, seed=2)  # excluded
    org2 = _survival_df(3, seed=3)  # excluded

    result = _run_central(
        [org0, org1, org2], time_col="time", outcome_col="event", expl_vars=["age"],
        minimum_events=10, min_organizations=2,
    )
    assert "error" in result
    assert "min_organizations" in result["error"]
    assert "minimum_events" in result["error"]
    assert "mock-1" in result["error"]
    assert "mock-2" in result["error"]


def test_minimum_events_excludes_low_event_organizations_but_still_succeeds() -> None:
    org0 = _survival_df(20, seed=1)
    org1 = _survival_df(3, seed=2)  # excluded, min_organizations=1 so run still succeeds

    result = _run_central(
        [org0, org1], time_col="time", outcome_col="event", expl_vars=["age"],
        minimum_events=10, min_organizations=1,
    )
    assert "error" not in result
    assert result["included_organizations"] == [0]
    assert result["excluded_organizations"] == [1]
