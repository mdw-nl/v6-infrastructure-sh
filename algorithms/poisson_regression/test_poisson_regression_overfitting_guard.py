import numpy as np
import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="poisson_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _poisson_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    mu = np.exp(0.5 + 0.3 * x1 - 0.2 * x2)
    y = rng.poisson(mu)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y.astype(float)})


def _base_kwargs(**overrides) -> dict:
    kwargs = {"predictor_cols": ["x1", "x2"], "outcome_col": "y"}
    kwargs.update(overrides)
    return kwargs


def test_max_pct_vars_vs_obs_validation_rejects_non_positive() -> None:
    result = _run_central([_poisson_df(20, 1)], **_base_kwargs(max_pct_vars_vs_obs=0))
    assert "max_pct_vars_vs_obs" in result["error"]


def test_max_pct_vars_vs_obs_errors_when_too_few_rows_per_parameter() -> None:
    # 20 rows, 3 parameters (intercept + 2 predictors) -> 15% ratio, which
    # exceeds a 5% threshold.
    result = _run_central([_poisson_df(20, 1)], **_base_kwargs(max_pct_vars_vs_obs=5))
    assert "error" in result
    assert "max_pct_vars_vs_obs" in result["error"]
    assert "3" in result["error"]
    assert "20" in result["error"]


def test_max_pct_vars_vs_obs_succeeds_when_ratio_within_threshold() -> None:
    result = _run_central([_poisson_df(20, 1)], **_base_kwargs(max_pct_vars_vs_obs=50))
    assert "error" not in result


def test_max_pct_vars_vs_obs_default_is_no_op() -> None:
    # Same thin ratio as the error case above, but without setting
    # max_pct_vars_vs_obs the run must proceed uninhibited.
    result = _run_central([_poisson_df(20, 1)], **_base_kwargs())
    assert "error" not in result


def test_max_pct_vars_vs_obs_is_checked_against_rows_after_min_rows_exclusion() -> None:
    # org1's 5 rows don't clear min_rows=10, so they must not count toward
    # the parameters-per-row denominator: only org0's 20 rows should be
    # used, which clears a 15% threshold for 3 parameters.
    org0 = _poisson_df(20, 1)  # qualifies
    org1 = _poisson_df(5, 2)  # excluded
    result = _run_central(
        [org0, org1], **_base_kwargs(min_rows=10, min_organizations=1, max_pct_vars_vs_obs=15)
    )
    assert "error" not in result
