import numpy as np
import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

FEATURE_COLS = ["x0", "x1"]
TARGET_COL = "y"


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="admm_logistic_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _separable_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    y = (x0 + x1 > 0).astype(int)
    return pd.DataFrame({"x0": x0, "x1": x1, "y": y})


def _base_kwargs(**overrides) -> dict:
    kwargs = {"feature_cols": FEATURE_COLS, "target_col": TARGET_COL, "num_rounds": 2}
    kwargs.update(overrides)
    return kwargs


def test_min_organizations_validation_rejects_less_than_one() -> None:
    result = _run_central([_separable_df(30, 1)], **_base_kwargs(min_organizations=0))
    assert "min_organizations" in result["error"]


def test_min_rows_validation_rejects_negative() -> None:
    result = _run_central([_separable_df(30, 1)], **_base_kwargs(min_rows=-1))
    assert "min_rows" in result["error"]


def test_min_organizations_fails_fast_when_task_has_too_few_orgs() -> None:
    org0 = _separable_df(30, 1)
    org1 = _separable_df(30, 2)
    result = _run_central(
        [org0, org1], **_base_kwargs(organizations=["mock-0"], min_organizations=2)
    )
    assert "error" in result
    assert "min_organizations" in result["error"]


def test_min_rows_and_min_organizations_together_requires_qualifying_orgs() -> None:
    # 3 organizations are nominally included -- enough to satisfy
    # min_organizations=2 on its own -- but only one has enough rows to
    # clear min_rows=20. Having enough organizations included is not the
    # same as having enough organizations with enough data.
    org0 = _separable_df(30, 1)  # qualifies
    org1 = _separable_df(5, 2)  # excluded
    org2 = _separable_df(5, 3)  # excluded

    result = _run_central(
        [org0, org1, org2], **_base_kwargs(min_rows=20, min_organizations=2)
    )
    assert "error" in result
    assert "min_organizations" in result["error"]
    assert "min_rows" in result["error"]
    assert "mock-1" in result["error"]
    assert "mock-2" in result["error"]


def test_min_rows_excludes_low_row_organizations_but_still_succeeds() -> None:
    org0 = _separable_df(30, 1)
    org1 = _separable_df(5, 2)  # excluded, min_organizations=1 so run still succeeds

    result = _run_central([org0, org1], **_base_kwargs(min_rows=20, min_organizations=1))
    assert "error" not in result
