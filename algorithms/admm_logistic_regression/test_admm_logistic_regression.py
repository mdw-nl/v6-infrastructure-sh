import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from admm_logistic_regression import (
    _logistic_admm_objective,
    _predict_proba,
    _split_train_val,
    _z_objective,
)

FEATURE_COLS = ["x0", "x1"]
TARGET_COL = "y"


# ── _logistic_admm_objective ──────────────────────────────────────────────────


def _numeric_gradient(x, z, u, rho, features, outcome, total_patients, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus, _ = _logistic_admm_objective(x_plus, z, u, rho, features, outcome, total_patients)
        f_minus, _ = _logistic_admm_objective(x_minus, z, u, rho, features, outcome, total_patients)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def test_admm_objective_analytic_gradient_matches_numeric_gradient() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 3))
    outcome = rng.integers(0, 2, size=20).astype(float)
    x = rng.normal(scale=0.5, size=4)  # +1 for intercept
    z = np.zeros(4)
    u = np.zeros(4)

    _, analytic_grad = _logistic_admm_objective(x, z, u, rho=1.0, features=features, outcome=outcome, total_patients=20)
    numeric_grad = _numeric_gradient(x, z, u, 1.0, features, outcome, 20)

    assert np.allclose(analytic_grad, numeric_grad, atol=1e-4)


def test_admm_objective_gradient_has_no_nan_or_inf_for_large_x() -> None:
    # Regression test: the value computation clips logits to [-500, 500]
    # before exp(), but the gradient loop's exp_term used to be unclipped,
    # so a large x could overflow to inf -> nan in the gradient.
    features = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
    outcome = np.array([1.0, 0.0, 1.0])
    x = np.array([300.0, 300.0, 300.0])  # xi @ x reaches ~900 for some rows, past float64 exp overflow (~709)
    z = np.zeros(3)
    u = np.zeros(3)

    value, grad = _logistic_admm_objective(x, z, u, rho=1.0, features=features, outcome=outcome, total_patients=3)

    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))


# ── _z_objective ───────────────────────────────────────────────────────────────


def test_z_objective_gradient_has_no_nan_when_a_component_is_exactly_zero() -> None:
    # Regression test: abs_z was only floored at index 0 (the intercept), so
    # any other component landing on exactly 0.0 produced a 0/0 = NaN in the
    # L1 gradient term, poisoning the whole gradient even when lambda_ > 0.
    z = np.array([1.0, 0.0, 2.0])  # non-intercept component (index 1) is exactly zero
    x_hat_mean = np.zeros(3)
    u_mean = np.zeros(3)

    value, grad = _z_objective(z, x_hat_mean, u_mean, rho=1.0, lambda_=0.5, num_sites=2)

    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))


def test_z_objective_gradient_matches_numeric_gradient_away_from_zero() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=4)
    z[0] = 0.7  # keep away from exact zero so the numeric check isn't near the L1 kink
    x_hat_mean = rng.normal(size=4)
    u_mean = rng.normal(size=4)
    rho, lambda_, num_sites = 0.8, 0.3, 3

    _, analytic_grad = _z_objective(z, x_hat_mean, u_mean, rho, lambda_, num_sites)

    eps = 1e-6
    numeric_grad = np.zeros_like(z)
    for i in range(len(z)):
        z_plus, z_minus = z.copy(), z.copy()
        z_plus[i] += eps
        z_minus[i] -= eps
        f_plus, _ = _z_objective(z_plus, x_hat_mean, u_mean, rho, lambda_, num_sites)
        f_minus, _ = _z_objective(z_minus, x_hat_mean, u_mean, rho, lambda_, num_sites)
        numeric_grad[i] = (f_plus - f_minus) / (2 * eps)

    assert np.allclose(analytic_grad, numeric_grad, atol=1e-3)


# ── _split_train_val ────────────────────────────────────────────────────────────


def test_split_train_val_respects_train_ratio_and_is_seeded_deterministically() -> None:
    df = pd.DataFrame({"x0": range(20), "x1": range(20), "y": [0, 1] * 10})
    train_a, val_a = _split_train_val(df, FEATURE_COLS, TARGET_COL, train_ratio=0.5, seed=7)
    train_b, val_b = _split_train_val(df, FEATURE_COLS, TARGET_COL, train_ratio=0.5, seed=7)

    assert len(train_a) == 10
    assert len(val_a) == 10
    assert train_a.index.tolist() == train_b.index.tolist()
    assert set(train_a.index) & set(val_a.index) == set()


def test_split_train_val_drops_rows_with_missing_feature_or_target() -> None:
    df = pd.DataFrame({"x0": [1.0, None, 3.0], "x1": [1.0, 2.0, None], "y": [0, 1, 0]})
    train_df, val_df = _split_train_val(df, FEATURE_COLS, TARGET_COL, train_ratio=1.0, seed=1)
    assert len(train_df) == 1
    assert len(val_df) == 0


# ── generic synthetic data helpers ─────────────────────────────────────────────


def _classification_df(n: int, seed: int) -> pd.DataFrame:
    # x1 is deliberately on a much larger scale than x0, so a test using both
    # columns exercises the federated normalization phase, not just the ADMM
    # math -- if normalization were skipped or applied inconsistently between
    # training/evaluation/prediction, this scale mismatch would show up as a
    # much worse probability match in the pooled-equivalence test below.
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 1.0, size=n)
    x1 = rng.normal(0.0, 50.0, size=n)
    logits = 1.2 * x0 - 0.03 * x1
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)
    return pd.DataFrame({"x0": x0, "x1": x1, "y": y})


def _call_node_method(df: pd.DataFrame, method: str, **kwargs) -> dict:
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="admm_logistic_regression")
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": method, "kwargs": kwargs}, organizations=org_ids)
    return client.wait_for_results(task.get("id"))[0]


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="admm_logistic_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


# ── compute_stats ──────────────────────────────────────────────────────────────


def test_compute_stats_basic_sum_and_count() -> None:
    df = pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0, 5.0], "x1": [10.0, 20.0, 30.0, 40.0, 50.0], "y": [0, 1, 0, 1, 0]})
    result = _call_node_method(df, "compute_stats", feature_cols=FEATURE_COLS, target_col=TARGET_COL, train_ratio=1.0, seed=1)
    assert result["n"] == 5
    assert result["sum"] == pytest.approx([15.0, 150.0])
    assert result["sum_sq"] == pytest.approx([55.0, 5500.0])


def test_compute_stats_drops_rows_with_missing_feature_or_target() -> None:
    df = pd.DataFrame({"x0": [1.0, None, 3.0], "x1": [1.0, 2.0, 3.0], "y": [0, 1, None]})
    result = _call_node_method(df, "compute_stats", feature_cols=FEATURE_COLS, target_col=TARGET_COL, train_ratio=1.0, seed=1)
    assert result["n"] == 1
    assert result["sum"] == pytest.approx([1.0, 1.0])


def test_compute_stats_all_missing_returns_empty() -> None:
    df = pd.DataFrame({"x0": [None, None], "x1": [None, None], "y": [None, None]})
    result = _call_node_method(df, "compute_stats", feature_cols=FEATURE_COLS, target_col=TARGET_COL)
    assert result == {"n": 0, "sum": [0.0, 0.0], "sum_sq": [0.0, 0.0]}


# ── column validation ──────────────────────────────────────────────────────────


def test_compute_stats_raises_clear_error_for_missing_column() -> None:
    df = pd.DataFrame({"x0": [1.0, 2.0], "y": [0, 1]})
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="admm_logistic_regression")
    org_ids = [organization["id"] for organization in client.organization.list()]
    with pytest.raises(ValueError, match="x1"):
        client.task.create(
            input_={"method": "compute_stats", "kwargs": {"feature_cols": FEATURE_COLS, "target_col": TARGET_COL}},
            organizations=org_ids,
        )


# ── node-side functions (admm_x_update_partial / evaluate_global_model / collect_predictions) ──


def test_admm_x_update_partial_returns_a_coefficient_vector_of_the_right_length() -> None:
    df = _classification_df(80, seed=1)
    num_features = len(FEATURE_COLS) + 1  # +1 intercept
    result = _call_node_method(
        df,
        "admm_x_update_partial",
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        z=[0.0] * num_features,
        u=[0.0] * num_features,
        rho=0.25,
        total_patients=80,
        global_mean=[0.0, 0.0],
        global_std=[1.0, 50.0],
        x_prev=[0.0] * num_features,
    )
    assert len(result["x"]) == num_features
    assert result["patient_count"] <= 80
    assert 0.0 <= result["train_acc"] <= 1.0


def test_evaluate_global_model_reports_validation_metrics() -> None:
    df = _classification_df(80, seed=2)
    num_features = len(FEATURE_COLS) + 1
    _, val_df = _split_train_val(df, FEATURE_COLS, TARGET_COL, train_ratio=0.8, seed=42)
    result = _call_node_method(
        df,
        "evaluate_global_model",
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        z=[0.0] * num_features,
        global_mean=[0.0, 0.0],
        global_std=[1.0, 50.0],
    )
    assert result["val_patient_count"] == len(val_df)
    assert 0.0 <= result["val_acc"] <= 1.0


def test_collect_predictions_returns_validation_set_only() -> None:
    df = _classification_df(80, seed=3)
    num_features = len(FEATURE_COLS) + 1
    _, val_df = _split_train_val(df, FEATURE_COLS, TARGET_COL, train_ratio=0.8, seed=42)
    result = _call_node_method(
        df,
        "collect_predictions",
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        z=[0.0] * num_features,
        global_mean=[0.0, 0.0],
        global_std=[1.0, 50.0],
    )
    assert len(result["y_all"]) == len(val_df)
    assert len(result["probs_all"]) == len(val_df)


def test_normalization_is_applied_consistently_in_eval_and_predict() -> None:
    # Regression guard for the single biggest risk in adding normalization:
    # if evaluate_global_model or collect_predictions forgot to apply
    # global_mean/global_std (or applied it differently), this catches it.
    # Both are deterministic given z (no re-optimization like
    # admm_x_update_partial, which would mask a missed normalization step by
    # just fitting around it) -- a z vector with a steep boundary in
    # NORMALIZED x0-space only classifies correctly if both functions
    # normalize x0 by the same (mean, std).
    rng = np.random.default_rng(5)
    x0 = rng.normal(10.0, 2.0, size=60)  # raw mean=10, std=2 -- normalization matters here
    y = (x0 > 10.0).astype(int)
    df = pd.DataFrame({"x0": x0, "x1": rng.normal(size=60), "y": y})

    stats = _call_node_method(df, "compute_stats", feature_cols=FEATURE_COLS, target_col=TARGET_COL, train_ratio=1.0, seed=1)
    mean = [stats["sum"][0] / stats["n"], stats["sum"][1] / stats["n"]]
    var = [stats["sum_sq"][i] / stats["n"] - mean[i] ** 2 for i in range(2)]
    std = [float(np.sqrt(max(v, 1e-8))) for v in var]

    # Decision rule in NORMALIZED space: predict 1 iff normalized x0 > 0,
    # i.e. iff raw x0 > mean -- exactly the rule y was generated from.
    # train_ratio=0.0 puts the whole set into validation.
    z = [0.0, 10.0, 0.0]  # intercept, x0 weight, x1 weight

    eval_result = _call_node_method(
        df, "evaluate_global_model", feature_cols=FEATURE_COLS, target_col=TARGET_COL,
        z=z, global_mean=mean, global_std=std, train_ratio=0.0,
    )
    assert eval_result["val_acc"] > 0.9

    pred_result = _call_node_method(
        df, "collect_predictions", feature_cols=FEATURE_COLS, target_col=TARGET_COL,
        z=z, global_mean=mean, global_std=std, train_ratio=0.0,
    )
    predicted = (np.array(pred_result["probs_all"]) >= 0.5).astype(int)
    assert np.mean(predicted == np.array(pred_result["y_all"])) > 0.9


# ── central: full pipeline ──────────────────────────────────────────────────────


def test_central_returns_error_when_no_usable_data() -> None:
    df = pd.DataFrame({"x0": [None], "x1": [None], "y": [None]})
    result = _run_central([df], feature_cols=FEATURE_COLS, target_col=TARGET_COL)
    assert result == {"error": "No usable data across any organization"}


def test_central_basic_shape() -> None:
    df = _classification_df(60, seed=6)
    result = _run_central([df], feature_cols=FEATURE_COLS, target_col=TARGET_COL, num_rounds=3)
    assert set(result) == {
        "coefficients", "feature_cols", "global_mean", "global_std", "history",
        "rho", "lambda_", "alpha", "abs_tol", "rel_tol", "patient_counts",
        "roc_site", "roc_global", "calibration",
    }
    assert result["feature_cols"] == FEATURE_COLS
    assert len(result["coefficients"]) == len(FEATURE_COLS) + 1


def test_central_global_normalization_matches_pooled_computation() -> None:
    # Phase 1 (global_mean/global_std) is pure sum/count aggregation, not
    # iterative optimization -- it must match pooling all nodes' training
    # rows exactly, the same property tested for logistic_regression.py.
    node_a = _classification_df(70, seed=7)
    node_b = _classification_df(30, seed=8)
    result = _run_central([node_a, node_b], feature_cols=FEATURE_COLS, target_col=TARGET_COL, num_rounds=1)

    train_a, _ = _split_train_val(node_a, FEATURE_COLS, TARGET_COL, train_ratio=0.8, seed=42)
    train_b, _ = _split_train_val(node_b, FEATURE_COLS, TARGET_COL, train_ratio=0.8, seed=42)
    pooled_train = pd.concat([train_a, train_b])

    for i, col in enumerate(FEATURE_COLS):
        assert result["global_mean"][i] == pytest.approx(pooled_train[col].mean(), rel=1e-5)
        variance = (pooled_train[col] ** 2).mean() - pooled_train[col].mean() ** 2
        assert result["global_std"][i] == pytest.approx(np.sqrt(max(variance, 1e-8)), rel=1e-5)


def test_admm_approximately_matches_pooled_logistic_regression_predictions() -> None:
    # Run the federated ADMM procedure split across several organizations and
    # compare its consensus model's predicted probabilities to a directly-fit
    # centralized logistic regression on the pooled (and identically
    # normalized) training data.
    #
    # This is not exact coefficient recovery: what matters is that ADMM's
    # predicted probabilities on the pooled training set are close to the
    # pooled reference fit's -- together with
    # test_admm_history_shows_decreasing_primal_residual (which separately
    # confirms r_norm/s_norm decrease monotonically), this is evidence ADMM is
    # converging toward the pooled solution rather than to some other fixed
    # point.
    node_dfs = [_classification_df(200, seed=s) for s in (10, 11, 12)]
    pooled = pd.concat(node_dfs, ignore_index=True)

    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in node_dfs],
        module="admm_logistic_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "feature_cols": FEATURE_COLS,
                "target_col": TARGET_COL,
                "num_rounds": 150,
                "rho": 0.25,
                "alpha": 1,
                "lambda_": 0.0,
                "abs_tol": 1e-8,
                "rel_tol": 1e-8,
            },
        },
        organizations=[org_ids[0]],
    )
    result = client.wait_for_results(task.get("id"))[0]
    admm_coef = np.array(result["coefficients"])
    mean = np.array(result["global_mean"])
    std = np.array(result["global_std"])

    train_pooled, _ = _split_train_val(pooled, FEATURE_COLS, TARGET_COL, train_ratio=0.8, seed=42)
    X_normalized = (train_pooled[FEATURE_COLS].to_numpy(dtype=float) - mean) / std
    y_train = train_pooled[TARGET_COL].to_numpy(dtype=float)

    reference = LogisticRegression(C=np.inf, max_iter=2000, tol=1e-10)
    reference.fit(X_normalized, y_train)

    admm_probs = _predict_proba(X_normalized, admm_coef)
    reference_probs = reference.predict_proba(X_normalized)[:, 1]
    assert np.mean(np.abs(admm_probs - reference_probs)) < 0.05


def test_admm_history_shows_decreasing_primal_residual() -> None:
    node_dfs = [_classification_df(60, seed=s) for s in (20, 21, 22)]
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in node_dfs],
        module="admm_logistic_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "feature_cols": FEATURE_COLS, "target_col": TARGET_COL,
                "num_rounds": 15, "rho": 0.25, "alpha": 1, "lambda_": 0.0,
            },
        },
        organizations=[org_ids[0]],
    )
    result = client.wait_for_results(task.get("id"))[0]
    r_norm_history = result["history"]["r_norm"]
    # Later rounds should show a smaller primal residual than the first
    # round -- evidence the ADMM iterations are actually converging, not
    # just running a fixed number of times.
    assert r_norm_history[-1] < r_norm_history[0]


def test_admm_does_not_crash_when_no_node_has_validation_patients() -> None:
    # Regression test: train_ratio=1.0 means every row on every node is
    # training data, so total_val_patients across the whole collaboration is
    # 0. This should report NaN val_rmse for every round (not raise
    # ZeroDivisionError) and skip ROC/AUC entirely (not raise from sklearn's
    # roc_curve on an empty y array).
    node_dfs = [_classification_df(30, seed=s) for s in (30, 31, 32)]
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in node_dfs],
        module="admm_logistic_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(
        input_={
            "method": "central",
            "kwargs": {
                "feature_cols": FEATURE_COLS, "target_col": TARGET_COL,
                "num_rounds": 3, "rho": 0.25, "alpha": 1, "lambda_": 0.0,
                "train_ratio": 1.0,
            },
        },
        organizations=[org_ids[0]],
    )
    result = client.wait_for_results(task.get("id"))[0]
    assert all(np.isnan(v) for v in result["history"]["val_rmse"])
    assert result["roc_site"] == []
    assert result["roc_global"] == {}
