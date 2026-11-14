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
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time": rng.integers(1, 200, size=n_events).astype(float),
            "event": [1] * n_events,
            "age": 40 + rng.integers(0, 50, size=n_events),
            "cov2": rng.normal(size=n_events),
        }
    )


def test_max_pct_vars_vs_obs_validation_rejects_non_positive() -> None:
    df = _survival_df(20, seed=1)
    result = _run_central(
        [df], time_col="time", outcome_col="event", expl_vars=["age"], max_pct_vars_vs_obs=0,
    )
    assert "max_pct_vars_vs_obs" in result["error"]


def test_max_pct_vars_vs_obs_errors_when_too_few_events_per_covariate() -> None:
    # 20 events, 2 covariates -> 10% events-per-covariate ratio, which
    # exceeds a 5% threshold.
    df = _survival_df(20, seed=1)
    result = _run_central(
        [df], time_col="time", outcome_col="event", expl_vars=["age", "cov2"],
        max_pct_vars_vs_obs=5,
    )
    assert "error" in result
    assert "max_pct_vars_vs_obs" in result["error"]
    assert "2" in result["error"]
    assert "20" in result["error"]


def test_max_pct_vars_vs_obs_succeeds_when_ratio_within_threshold() -> None:
    df = _survival_df(20, seed=1)
    result = _run_central(
        [df], time_col="time", outcome_col="event", expl_vars=["age", "cov2"],
        max_pct_vars_vs_obs=50,
    )
    assert "error" not in result


def test_max_pct_vars_vs_obs_default_is_no_op() -> None:
    # Same thin ratio as the error case above, but without setting
    # max_pct_vars_vs_obs the run must proceed uninhibited.
    df = _survival_df(20, seed=1)
    result = _run_central(
        [df], time_col="time", outcome_col="event", expl_vars=["age", "cov2"],
    )
    assert "error" not in result


def test_max_pct_vars_vs_obs_is_checked_against_events_after_minimum_events_exclusion() -> None:
    # org1's 3 events don't clear minimum_events=10, so they must not count
    # toward the events-per-covariate denominator: only org0's 20 events
    # should be used, which clears a 10% threshold for 2 covariates.
    org0 = _survival_df(20, seed=1)
    org1 = _survival_df(3, seed=2)
    result = _run_central(
        [org0, org1], time_col="time", outcome_col="event", expl_vars=["age", "cov2"],
        minimum_events=10, min_organizations=1, max_pct_vars_vs_obs=10,
    )
    assert "error" not in result
    assert result["included_organizations"] == [0]
