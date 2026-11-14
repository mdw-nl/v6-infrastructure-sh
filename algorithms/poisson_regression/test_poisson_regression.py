import json

import numpy as np
import pandas as pd
import pytest
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from poisson_regression import _filter_organizations, _require_columns


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="poisson_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _poisson_df(n: int, seed: int, true_beta: tuple = (0.5, 0.3, -0.2)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = true_beta[0] + true_beta[1] * x1 + true_beta[2] * x2
    mu = np.exp(eta)
    y = rng.poisson(mu)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y.astype(float)})


def _reference_irls_fit(df: pd.DataFrame, max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Independent (non-federated, non-split) IRLS fit on pooled data, used
    as ground truth -- deliberately re-implemented here rather than calling
    into poisson_regression.py, the same way coxph's test_math_correctness.py
    re-implements its own reference optimizer instead of trusting coxph.py."""
    X = np.column_stack([np.ones(len(df)), df["x1"].to_numpy(), df["x2"].to_numpy()])
    y = df["y"].to_numpy(dtype=float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-6))
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        W = mu
        z = eta + (y - mu) / mu
        XtWX = (X * W[:, None]).T @ X
        XtWz = (X * W[:, None]).T @ z
        beta_new = np.linalg.solve(XtWX, XtWz)
        delta = np.max(np.abs(beta_new - beta))
        beta = beta_new
        if delta < tol:
            break
    return beta


def test_federated_fit_matches_pooled_reference_fit_across_multiple_organizations() -> None:
    df = _poisson_df(400, seed=1)
    ref_beta = _reference_irls_fit(df)

    splits = np.array_split(df, 4)
    result = _run_central(list(splits), predictor_cols=["x1", "x2"], outcome_col="y")

    assert result["converged"]
    model = json.loads(result["model"])
    fed_beta = np.array([model["Coef"]["Intercept"], model["Coef"]["x1"], model["Coef"]["x2"]])
    assert np.allclose(fed_beta, ref_beta, atol=1e-4)
    assert result["num_observations"] == 400


def test_overall_p_value_is_significant_for_a_strong_real_relationship() -> None:
    df = _poisson_df(400, seed=2, true_beta=(0.5, 0.8, -0.6))
    result = _run_central([df], predictor_cols=["x1", "x2"], outcome_col="y")
    assert result["overall_p_value"] < 0.001


def test_overall_p_value_is_not_significant_for_pure_noise() -> None:
    rng = np.random.default_rng(3)
    n = 150
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "y": rng.poisson(2.0, size=n).astype(float),
        }
    )
    result = _run_central([df], predictor_cols=["x1", "x2"], outcome_col="y")
    assert result["overall_p_value"] > 0.05


def test_negative_outcome_raises_clear_error() -> None:
    df = _poisson_df(50, seed=4)
    df.loc[0, "y"] = -1.0
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="poisson_regression")
    org_ids = [organization["id"] for organization in client.organization.list()]
    with pytest.raises((ValueError, SystemExit)):
        client.task.create(
            input_={"method": "compute_local_row_stats", "kwargs": {"predictor_cols": ["x1", "x2"], "outcome_col": "y"}},
            organizations=[org_ids[0]],
        )


def test_non_integer_outcome_raises_clear_error() -> None:
    df = _poisson_df(50, seed=5)
    df.loc[0, "y"] = 2.5
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="poisson_regression")
    org_ids = [organization["id"] for organization in client.organization.list()]
    with pytest.raises((ValueError, SystemExit)):
        client.task.create(
            input_={"method": "compute_local_row_stats", "kwargs": {"predictor_cols": ["x1", "x2"], "outcome_col": "y"}},
            organizations=[org_ids[0]],
        )


def test_missing_column_raises_clear_error() -> None:
    client = MockAlgorithmClient(
        datasets=[[{"database": _poisson_df(50, seed=6), "input_data": {}}]],
        module="poisson_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    with pytest.raises((ValueError, SystemExit)):
        client.task.create(
            input_={
                "method": "compute_local_row_stats",
                "kwargs": {"predictor_cols": ["x1", "does_not_exist"], "outcome_col": "y"},
            },
            organizations=[org_ids[0]],
        )


def test_perfectly_collinear_predictors_return_a_clear_error_instead_of_garbage() -> None:
    df = _poisson_df(100, seed=7)
    df["x1_copy"] = df["x1"]
    result = _run_central([df], predictor_cols=["x1", "x1_copy"], outcome_col="y")
    assert "error" in result
    assert "rank-deficient" in result["error"]


def test_not_enough_rows_for_residual_degrees_of_freedom_returns_clear_error() -> None:
    df = _poisson_df(3, seed=8)
    result = _run_central([df], predictor_cols=["x1", "x2"], outcome_col="y")
    assert "error" in result
    assert "residual degrees of freedom" in result["error"]


def test_dispersion_estimate_is_reasonable_for_true_poisson_data() -> None:
    df = _poisson_df(400, seed=9)
    result = _run_central([df], predictor_cols=["x1", "x2"], outcome_col="y")
    # True Poisson-distributed data should show no strong over/under-dispersion.
    assert 0.7 < result["dispersion_estimate"] < 1.3
    assert not any("overdispersed" in w for w in result["warnings"])


def test_overdispersed_data_triggers_a_warning() -> None:
    # A negative-binomial-like mixture has variance >> mean, violating the
    # Poisson assumption -- dispersion_estimate should be well above 1 and
    # central() should warn about it.
    rng = np.random.default_rng(10)
    n = 300
    x1 = rng.normal(size=n)
    gamma_noise = rng.gamma(shape=0.5, scale=2.0, size=n)
    mu = np.exp(0.5 + 0.3 * x1) * gamma_noise
    y = rng.poisson(mu)
    df = pd.DataFrame({"x1": x1, "y": y.astype(float)})
    result = _run_central([df], predictor_cols=["x1"], outcome_col="y")
    assert result["dispersion_estimate"] > 1.5
    assert any("overdispersed" in w for w in result["warnings"])


def test_filter_organizations_rejects_unknown_name() -> None:
    orgs = [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]
    with pytest.raises(ValueError, match="not found"):
        _filter_organizations(orgs, ["gamma"], "central")


def test_require_columns_rejects_missing_column() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="not found in dataset"):
        _require_columns(df, ["a", "b"], "compute_local_row_stats")


def test_duplicate_predictor_cols_returns_a_clear_error() -> None:
    # A duplicate name here is not the same thing as two distinct but
    # collinear columns (see the collinearity test above) -- it's caught
    # separately, before it can reach pandas' column-selection quirk where
    # indexing an already-duplicated label multiplies columns out instead
    # of just repeating them.
    df = _poisson_df(50, seed=11)
    result = _run_central([df], predictor_cols=["x1", "x1"], outcome_col="y")
    assert "error" in result
    assert "duplicate column name" in result["error"]


def test_outcome_col_in_predictor_cols_returns_a_clear_error() -> None:
    df = _poisson_df(50, seed=12)
    result = _run_central([df], predictor_cols=["x1", "y"], outcome_col="y")
    assert "error" in result
    assert "must not also appear" in result["error"]
