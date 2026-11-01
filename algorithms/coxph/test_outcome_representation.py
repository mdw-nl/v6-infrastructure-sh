import pandas as pd
import pytest
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

from coxph import _derive_binary


def _call_partial(df: pd.DataFrame, method: str, **kwargs) -> dict:
    client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="coxph")
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": method, "kwargs": kwargs}, organizations=org_ids)
    return client.wait_for_results(task.get("id"))[0]


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="coxph",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


# ── _derive_binary unit tests ─────────────────────────────────────────────────────


def test_derive_binary_identity_passthrough_when_already_binary() -> None:
    result = _derive_binary(pd.Series([0, 1, 0]), None, "event")
    assert result.tolist() == [0.0, 1.0, 0.0]


def test_derive_binary_presence_mode() -> None:
    series = pd.Series(["2020-01-01", None, "2021-06-15"])
    result = _derive_binary(series, {"mode": "presence"}, "death_date")
    assert result.tolist() == [1.0, 0.0, 1.0]


def test_derive_binary_equals_mode_preserves_missing() -> None:
    series = pd.Series(["Deceased", "Alive", None])
    result = _derive_binary(series, {"mode": "equals", "positive_value": "Deceased"}, "status")
    assert result.tolist()[:2] == [1.0, 0.0]
    assert pd.isna(result.iloc[2])


def test_derive_binary_isin_mode() -> None:
    series = pd.Series(["Dead", "Deceased", "Alive"])
    result = _derive_binary(series, {"mode": "isin", "positive_values": ["Dead", "Deceased"]}, "status")
    assert result.tolist() == [1.0, 1.0, 0.0]


def test_derive_binary_invert_flips_result() -> None:
    series = pd.Series(["2020-01-01", None])
    result = _derive_binary(series, {"mode": "presence", "invert": True}, "death_date")
    assert result.tolist() == [0.0, 1.0]


def test_derive_binary_raises_on_unknown_mode() -> None:
    with pytest.raises(ValueError, match=r"unknown representation mode"):
        _derive_binary(pd.Series([0, 1]), {"mode": "nonsense"}, "event")


def test_derive_binary_raises_when_result_is_not_binary() -> None:
    with pytest.raises(ValueError, match=r"non-binary value"):
        _derive_binary(pd.Series([0, 1, 2]), None, "event")


def test_derive_binary_raises_when_identity_column_is_not_numeric() -> None:
    with pytest.raises(ValueError, match=r"not numeric"):
        _derive_binary(pd.Series(["Deceased", "Alive"]), None, "status")


# ── node-method wiring ────────────────────────────────────────────────────────────


def test_get_unique_event_times_applies_outcome_representation() -> None:
    df = pd.DataFrame(
        {
            "Survival.time": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "death_date": ["2020-01-01"] * 11,
        }
    )
    result = _call_partial(
        df, "get_unique_event_times", time_col="Survival.time", outcome_col="death_date",
        outcome_representation={"mode": "presence"}, minimum_events=1,
    )
    assert "n_threshold_not_met" not in result
    assert len(result["times"]["Survival.time"]) == 11


def test_compute_summed_z_applies_outcome_representation() -> None:
    df = pd.DataFrame(
        {"age": [50.0, 60.0, 70.0], "status": ["Deceased", "Alive", "Deceased"]}
    )
    result = _call_partial(
        df, "compute_summed_z", outcome_col="status", expl_vars=["age"],
        outcome_representation={"mode": "equals", "positive_value": "Deceased"},
    )
    assert result["sum"] == {"age": 120.0}


def test_compute_summed_z_raises_clear_error_for_non_binary_outcome_without_representation() -> None:
    df = pd.DataFrame({"age": [50.0, 60.0], "status": ["Deceased", "Alive"]})
    with pytest.raises(ValueError, match=r"not numeric"):
        _call_partial(df, "compute_summed_z", outcome_col="status", expl_vars=["age"])


def test_central_end_to_end_with_presence_representation_matches_equivalent_binary_column() -> None:
    df_dates = pd.DataFrame(
        {
            "Survival.time": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            "age": [50, 55, 60, 65, 70, 75, 80, 85, 90, 52, 58, 62],
            "death_date": ["2020-01-01", None] * 6,
        }
    )
    df_binary = df_dates.assign(event=([1, 0] * 6)).drop(columns="death_date")

    result_dates = _run_central(
        [df_dates], time_col="Survival.time", outcome_col="death_date",
        outcome_representation={"mode": "presence"}, expl_vars=["age"],
    )
    result_binary = _run_central(
        [df_binary], time_col="Survival.time", outcome_col="event", expl_vars=["age"],
    )
    assert result_dates["model"] == result_binary["model"]
