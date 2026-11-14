import json

import numpy as np
import pandas as pd
import pytest
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="poisson_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _poisson_df_with_exposure(n: int, seed: int, true_beta: tuple = (0.5, 0.3, -0.2)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    exposure = rng.uniform(0.5, 5.0, size=n)
    eta = np.log(exposure) + true_beta[0] + true_beta[1] * x1 + true_beta[2] * x2
    mu = np.exp(eta)
    y = rng.poisson(mu)
    return pd.DataFrame({"x1": x1, "x2": x2, "exposure": exposure, "y": y.astype(float)})


def _reference_irls_fit_with_offset(df: pd.DataFrame, max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Independent (non-federated) IRLS-with-offset fit on pooled data --
    deliberately re-implemented here, not calling into poisson_regression.py,
    the same spirit as the no-offset reference fit in test_poisson_regression.py."""
    X = np.column_stack([np.ones(len(df)), df["x1"].to_numpy(), df["x2"].to_numpy()])
    y = df["y"].to_numpy(dtype=float)
    offset = np.log(df["exposure"].to_numpy(dtype=float))
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.sum() / df["exposure"].sum(), 1e-6))
    for _ in range(max_iter):
        eta = np.clip(offset + X @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        W = mu
        z = eta + (y - mu) / mu
        z_adjusted = z - offset
        XtWX = (X * W[:, None]).T @ X
        XtWz = (X * W[:, None]).T @ z_adjusted
        beta_new = np.linalg.solve(XtWX, XtWz)
        delta = np.max(np.abs(beta_new - beta))
        beta = beta_new
        if delta < tol:
            break
    return beta


def test_federated_fit_with_exposure_matches_pooled_reference_fit() -> None:
    df = _poisson_df_with_exposure(400, seed=1)
    ref_beta = _reference_irls_fit_with_offset(df)

    splits = np.array_split(df, 4)
    result = _run_central(
        list(splits), predictor_cols=["x1", "x2"], outcome_col="y", exposure_col="exposure"
    )

    assert result["converged"]
    model = json.loads(result["model"])
    fed_beta = np.array([model["Coef"]["Intercept"], model["Coef"]["x1"], model["Coef"]["x2"]])
    assert np.allclose(fed_beta, ref_beta, atol=1e-4)
    assert result["num_observations"] == 400


def test_omitting_exposure_col_is_equivalent_to_exposure_of_one_for_every_row() -> None:
    # A column of all-1s as the exposure is mathematically identical to no
    # offset at all (log(1) == 0) -- confirms the offset-aware code path
    # reduces exactly to the no-offset formulas rather than behaving
    # differently by coincidence.
    df = _poisson_df_with_exposure(300, seed=2)
    df["unit_exposure"] = 1.0

    with_unit_exposure = _run_central(
        [df], predictor_cols=["x1", "x2"], outcome_col="y", exposure_col="unit_exposure"
    )
    without_exposure = _run_central([df], predictor_cols=["x1", "x2"], outcome_col="y")

    model_a = json.loads(with_unit_exposure["model"])
    model_b = json.loads(without_exposure["model"])
    assert model_a["Coef"] == model_b["Coef"]
    assert np.isclose(with_unit_exposure["deviance"], without_exposure["deviance"])
    assert np.isclose(with_unit_exposure["null_deviance"], without_exposure["null_deviance"])
    assert np.isclose(with_unit_exposure["aic"], without_exposure["aic"])


def test_doubling_exposure_with_zero_coefficients_doubles_the_expected_rate() -> None:
    # Sanity/interpretability check: with predictors that don't matter
    # (coefficients near zero), the fitted rate per unit exposure should be
    # the same regardless of how much exposure each row has -- i.e. the
    # model correctly treats this as a rate, not a raw count, scaling
    # fitted mu proportionally with exposure for otherwise-identical rows.
    rng = np.random.default_rng(3)
    n = 500
    exposure = np.concatenate([np.full(n // 2, 1.0), np.full(n // 2, 2.0)])
    true_rate = 3.0
    y = rng.poisson(true_rate * exposure)
    df = pd.DataFrame({"x1": rng.normal(size=n), "exposure": exposure, "y": y.astype(float)})

    result = _run_central([df], predictor_cols=["x1"], outcome_col="y", exposure_col="exposure")
    model = json.loads(result["model"])
    fitted_rate = np.exp(model["Coef"]["Intercept"])
    assert abs(fitted_rate - true_rate) / true_rate < 0.15


def test_non_positive_exposure_raises_clear_error() -> None:
    # This validation lives inside compute_local_row_stats (a sub-task
    # central() dispatches), not in central()'s own upfront validation, so
    # it surfaces as a raised exception through MockAlgorithmClient's
    # synchronous execution model rather than a returned {"error": ...}.
    df = _poisson_df_with_exposure(50, seed=4)
    df.loc[0, "exposure"] = 0.0
    with pytest.raises((ValueError, SystemExit)):
        _run_central([df], predictor_cols=["x1", "x2"], outcome_col="y", exposure_col="exposure")


def test_exposure_col_overlapping_predictor_cols_returns_clear_error() -> None:
    df = _poisson_df_with_exposure(50, seed=5)
    result = _run_central(
        [df], predictor_cols=["x1", "exposure"], outcome_col="y", exposure_col="exposure"
    )
    assert "error" in result
    assert "must not also appear" in result["error"]


def test_exposure_col_same_as_outcome_col_returns_clear_error() -> None:
    df = _poisson_df_with_exposure(50, seed=6)
    result = _run_central(
        [df], predictor_cols=["x1", "x2"], outcome_col="y", exposure_col="y"
    )
    assert "error" in result
    assert "must not be the same column" in result["error"]
