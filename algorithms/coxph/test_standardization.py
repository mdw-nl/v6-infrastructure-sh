from io import StringIO

import numpy as np
import pandas as pd
import pytest
from coxph import compute_summed_z, perform_iteration
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="coxph",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _mock_client() -> MockAlgorithmClient:
    # A direct (non-task.create) call to a @algorithm_client-decorated
    # partial still needs a mock_client, or it tries to build a real
    # AlgorithmClient from unset HOST/PORT/... env vars.
    return MockAlgorithmClient(datasets=[[{"database": pd.DataFrame(), "input_data": {}}]], module="coxph")


def _survival_df(n: int, age_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "time": rng.integers(1, 200, size=n).astype(float),
            "event": rng.integers(0, 2, size=n),
            "age": (40 + rng.integers(0, 50, size=n)) * age_scale,
        }
    )


def test_compute_summed_z_reports_sum_squares_and_count_per_covariate_independently() -> None:
    # "height" is missing on one event row; its count must reflect that,
    # independently of "age" (which has no missing values among event rows).
    df = pd.DataFrame(
        {
            "event": [1, 1, 1, 0],
            "age": [50.0, 60.0, 70.0, 999.0],
            "height": [170.0, None, 180.0, 999.0],
        }
    )
    result = compute_summed_z(
        mock_data=[df], mock_client=_mock_client(), outcome_col="event", expl_vars=["age", "height"]
    )

    assert result["sum"] == {"age": 180.0, "height": 350.0}
    assert result["sum_squares"] == {"age": 50.0**2 + 60.0**2 + 70.0**2, "height": 170.0**2 + 180.0**2}
    assert result["count"] == {"age": 3.0, "height": 2.0}


def test_rescaling_a_covariate_leaves_the_fit_invariant_after_back_transform() -> None:
    # Fitting on age vs. age*2 (same information, different units) must
    # produce coefficients that differ by exactly the inverse scale factor,
    # and therefore identical hazard ratios per *original* unit -- this is
    # the property standardize-then-back-transform exists to preserve, and
    # it fails if the back-transform math (coxph.py's Phase 4) is wrong.
    # A modest factor (not e.g. 1000x) keeps both coefficients well above
    # the model table's own 5-decimal-place rounding granularity.
    df_natural = _survival_df(60, age_scale=1.0)
    df_rescaled = df_natural.assign(age=df_natural["age"] * 2.0)

    result_natural = _run_central([df_natural], time_col="time", outcome_col="event", expl_vars=["age"])
    result_rescaled = _run_central([df_rescaled], time_col="time", outcome_col="event", expl_vars=["age"])

    coef_natural = pd.read_json(StringIO(result_natural["model"]))["Coef"].iloc[0]
    coef_rescaled = pd.read_json(StringIO(result_rescaled["model"]))["Coef"].iloc[0]

    assert coef_rescaled == pytest.approx(coef_natural / 2.0, abs=2e-5)


def test_large_raw_covariate_scale_does_not_overflow() -> None:
    # Before standardization, a covariate on the order of 1e5 drove
    # exp(X @ beta) toward overflow as beta grew during iteration. This must
    # now converge to a finite, sane fit instead of NaN/inf coefficients.
    df = _survival_df(80, age_scale=5000.0)
    result = _run_central([df], time_col="time", outcome_col="event", expl_vars=["age"], max_iterations=25)

    model_df = pd.read_json(StringIO(result["model"]))
    assert np.isfinite(model_df["Coef"].to_numpy()).all()
    assert np.isfinite(model_df["SE"].to_numpy()).all()


def test_perform_iteration_without_centre_scale_falls_back_to_raw_covariates() -> None:
    # Direct/low-level calls that omit centre/scale (e.g. hand-written tests
    # of perform_iteration itself) must still work unstandardized.
    df = pd.DataFrame({"time": [10.0, 20.0, 30.0], "age": [40.0, 50.0, 60.0]})
    result = perform_iteration(
        mock_data=[df], mock_client=_mock_client(), time_col="time", expl_vars=["age"],
        beta=[0.0], unique_time_events=[10.0],
    )
    assert result["agg1"] == [3.0]


def test_perform_iteration_requires_both_centre_and_scale_together() -> None:
    df = pd.DataFrame({"time": [10.0, 20.0], "age": [40.0, 50.0]})
    with pytest.raises(ValueError, match="both"):
        perform_iteration(
            mock_data=[df], mock_client=_mock_client(), time_col="time", expl_vars=["age"],
            beta=[0.0], unique_time_events=[10.0], centre=[45.0], scale=None,
        )
