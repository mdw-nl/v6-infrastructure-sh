import numpy as np
import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="logistic_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _separable_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x > 0).astype(int)
    return pd.DataFrame({"x": x, "x2": x2, "y": y})


def _base_kwargs(**overrides) -> dict:
    # train_ratio=1.0 makes total training rows deterministically equal to
    # the dataset size, so the features-vs-rows ratio is exactly known.
    kwargs = {
        "feature_cols": ["x", "x2"], "target_col": "y", "n_rounds": 1,
        "local_epochs": 1, "train_ratio": 1.0,
    }
    kwargs.update(overrides)
    return kwargs


def test_max_pct_vars_vs_obs_validation_rejects_non_positive() -> None:
    result = _run_central([_separable_df(20, 1)], **_base_kwargs(max_pct_vars_vs_obs=0))
    assert "max_pct_vars_vs_obs" in result["error"]


def test_max_pct_vars_vs_obs_errors_when_too_few_rows_per_feature() -> None:
    # 20 training rows, 2 features -> 10% ratio, which exceeds a 5% threshold.
    result = _run_central([_separable_df(20, 1)], **_base_kwargs(max_pct_vars_vs_obs=5))
    assert "error" in result
    assert "max_pct_vars_vs_obs" in result["error"]
    assert "2" in result["error"]
    assert "20" in result["error"]


def test_max_pct_vars_vs_obs_succeeds_when_ratio_within_threshold() -> None:
    result = _run_central([_separable_df(20, 1)], **_base_kwargs(max_pct_vars_vs_obs=50))
    assert "error" not in result


def test_max_pct_vars_vs_obs_default_is_no_op() -> None:
    # Same thin ratio as the error case above, but without setting
    # max_pct_vars_vs_obs the run must proceed uninhibited.
    result = _run_central([_separable_df(20, 1)], **_base_kwargs())
    assert "error" not in result


def test_max_pct_vars_vs_obs_is_checked_against_rows_after_min_rows_exclusion() -> None:
    # org1's 5 rows don't clear min_rows=10, so they must not count toward
    # the features-per-row denominator: only org0's 20 rows should be used,
    # which clears a 10% threshold for 2 features.
    org0 = _separable_df(20, 1)  # qualifies
    org1 = _separable_df(5, 2)  # excluded
    result = _run_central(
        [org0, org1], **_base_kwargs(min_rows=10, min_organizations=1, max_pct_vars_vs_obs=10)
    )
    assert "error" not in result
