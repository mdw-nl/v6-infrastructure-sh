import json

import numpy as np
import pandas as pd
import pytest
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from linear_regression import _filter_organizations, _require_columns


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="linear_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _linear_df(n: int, seed: int, true_beta: tuple = (5.0, 2.0, -1.5), noise_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=noise_scale, size=n)
    y = true_beta[0] + true_beta[1] * x1 + true_beta[2] * x2 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _reference_fit(df: pd.DataFrame) -> tuple:
    X = np.column_stack([np.ones(len(df)), df["x1"].to_numpy(), df["x2"].to_numpy()])
    y = df["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ beta) ** 2))
    return beta, rss


def test_federated_fit_matches_pooled_reference_fit_across_multiple_organizations() -> None:
    df = _linear_df(300, seed=1)
    ref_beta, ref_rss = _reference_fit(df)

    splits = np.array_split(df, 4)
    result = _run_central(list(splits), predictor_cols=["x1", "x2"], outcome_col="y")

    model = json.loads(result["model"])
    fed_beta = np.array([model["Coef"]["Intercept"], model["Coef"]["x1"], model["Coef"]["x2"]])
    assert np.allclose(fed_beta, ref_beta, atol=1e-4)
    assert result["num_observations"] == 300
    assert result["degrees_of_freedom"] == 300 - 3


def test_r_squared_and_adjusted_r_squared_are_in_valid_range() -> None:
    df = _linear_df(200, seed=2)
    result = _run_central([df], predictor_cols=["x1", "x2"], outcome_col="y")
    assert 0.0 <= result["r_squared"] <= 1.0
    assert result["adjusted_r_squared"] <= result["r_squared"]


def test_overall_p_value_is_significant_for_a_strong_real_relationship() -> None:
    df = _linear_df(300, seed=3, true_beta=(5.0, 2.0, -1.5), noise_scale=0.5)
    result = _run_central([df], predictor_cols=["x1", "x2"], outcome_col="y")
    assert result["overall_p_value"] < 0.001


def test_overall_p_value_is_not_significant_for_pure_noise() -> None:
    rng = np.random.default_rng(4)
    n = 100
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "y": rng.normal(size=n)})
    result = _run_central([df], predictor_cols=["x1", "x2"], outcome_col="y")
    assert result["overall_p_value"] > 0.05


def test_perfectly_collinear_predictors_return_a_clear_error_instead_of_garbage() -> None:
    df = _linear_df(100, seed=5)
    df["x1_copy"] = df["x1"]  # perfectly collinear with x1
    result = _run_central([df], predictor_cols=["x1", "x1_copy"], outcome_col="y")
    assert "error" in result
    assert "rank-deficient" in result["error"]


def test_missing_column_raises_clear_error() -> None:
    client = MockAlgorithmClient(
        datasets=[[{"database": _linear_df(50, seed=6), "input_data": {}}]],
        module="linear_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    with pytest.raises((ValueError, SystemExit)):
        client.task.create(
            input_={
                "method": "compute_local_sums",
                "kwargs": {"predictor_cols": ["x1", "does_not_exist"], "outcome_col": "y"},
            },
            organizations=[org_ids[0]],
        )


def test_non_numeric_column_raises_clear_error() -> None:
    df = _linear_df(50, seed=7)
    df["x1"] = df["x1"].astype(str) + "not_a_number"
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="linear_regression")
    org_ids = [organization["id"] for organization in client.organization.list()]
    with pytest.raises((ValueError, SystemExit)):
        client.task.create(
            input_={"method": "compute_local_sums", "kwargs": {"predictor_cols": ["x1", "x2"], "outcome_col": "y"}},
            organizations=[org_ids[0]],
        )


def test_not_enough_rows_for_residual_degrees_of_freedom_returns_clear_error() -> None:
    # 3 parameters (intercept + 2 predictors), only 3 rows -> 0 residual
    # degrees of freedom, nothing to estimate variance from.
    df = _linear_df(3, seed=8)
    result = _run_central([df], predictor_cols=["x1", "x2"], outcome_col="y")
    assert "error" in result
    assert "residual degrees of freedom" in result["error"]


def test_compute_local_sums_matches_hand_computed_design_matrix() -> None:
    df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="linear_regression")
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={"method": "compute_local_sums", "kwargs": {"predictor_cols": ["x1"], "outcome_col": "y"}},
        organizations=[org_ids[0]],
    )
    out = client.wait_for_results(task.get("id"))[0]

    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([2.0, 4.0, 6.0])
    assert np.allclose(np.array(out["XtX"]), X.T @ X)
    assert np.allclose(np.array(out["XtY"]), X.T @ y)
    assert np.isclose(out["YtY"], float(np.dot(y, y)))
    assert out["n"] == 3


def test_filter_organizations_rejects_unknown_name() -> None:
    orgs = [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]
    with pytest.raises(ValueError, match="not found"):
        _filter_organizations(orgs, ["gamma"], "central")


def test_require_columns_rejects_missing_column() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="not found in dataset"):
        _require_columns(df, ["a", "b"], "compute_local_sums")
