import numpy as np
import pandas as pd
from coxph import apply_risk_set_jump_guard
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="coxph",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


# ── apply_risk_set_jump_guard (pure function) ────────────────────────────────


def test_default_min_change_of_one_never_holds() -> None:
    # Exactly one departure between consecutive releases is the normal case
    # for event-time grids (the event itself departs at the next time point)
    # -- min_change=1 must be a true no-op, not an always-on guard.
    agg1 = np.array([5.0, 4.0, 2.0])
    agg2 = np.array([[5.0], [4.0], [2.0]])
    agg3 = np.array([[[5.0]], [[4.0]], [[2.0]]])
    at_risk = np.array([5, 4, 2])

    out1, _, _, held = apply_risk_set_jump_guard(agg1, agg2, agg3, at_risk, min_change=1)
    assert not held.any()
    np.testing.assert_array_equal(out1, agg1)


def test_single_departure_is_held_when_min_change_requires_more() -> None:
    # 5 -> 4 at-risk is exactly one departure; min_change=2 must hold index 1
    # at index 0's values instead of releasing the fresh (thin) delta.
    agg1 = np.array([5.0, 4.0, 1.0])
    agg2 = np.array([[50.0], [40.0], [10.0]])
    agg3 = np.array([[[5.0]], [[4.0]], [[1.0]]])
    at_risk = np.array([5, 4, 1])  # deltas: 1, then 3

    out1, out2, _, held = apply_risk_set_jump_guard(agg1, agg2, agg3, at_risk, min_change=2)

    assert held.tolist() == [False, True, False]
    assert out1.tolist() == [5.0, 5.0, 1.0]  # index 1 held at index 0's value
    assert out2.tolist() == [[50.0], [50.0], [10.0]]
    # index 2 released fresh: 3 total departed since the last *release* (index 0,
    # since index 1 was held) -- 5 -> 1 is 4 departures, clears min_change=2.
    assert out1[2] == 1.0


def test_held_periods_accumulate_against_the_last_true_release() -> None:
    # Three single-departure steps in a row (5,4,3,2): with min_change=2,
    # every step after the first release must stay held until the *total*
    # departures since that release reaches 2, not reset after each hold.
    agg1 = np.array([5.0, 4.0, 3.0, 2.0])
    agg2 = np.array([[5.0], [4.0], [3.0], [2.0]])
    agg3 = np.array([[[5.0]], [[4.0]], [[3.0]], [[2.0]]])
    at_risk = np.array([5, 4, 3, 2])

    out1, _, _, held = apply_risk_set_jump_guard(agg1, agg2, agg3, at_risk, min_change=2)

    # index1: 5->4, 1 departed since release@0 -> held
    # index2: 5->3, 2 departed since release@0 -> released fresh (new release point)
    # index3: 3->2, 1 departed since release@2 -> held
    assert held.tolist() == [False, True, False, True]
    assert out1.tolist() == [5.0, 5.0, 3.0, 3.0]


def test_zero_time_points_is_a_no_op() -> None:
    empty = np.zeros(0)
    _, _, _, held = apply_risk_set_jump_guard(empty, empty.reshape(0, 1), empty.reshape(0, 1, 1), empty, 5)
    assert len(held) == 0


# ── end-to-end via central() ─────────────────────────────────────────────────


def _survival_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "time": rng.integers(1, 100, size=n).astype(float),
            "event": rng.integers(0, 2, size=n),
            "age": 40 + rng.integers(0, 50, size=n),
        }
    )


def test_strict_risk_set_min_change_still_converges_to_a_finite_fit() -> None:
    df = _survival_df(60)
    result = _run_central(
        [df], time_col="time", outcome_col="event", expl_vars=["age"],
        risk_set_min_change=3, max_iterations=25,
    )
    assert "error" not in result
    model_df = pd.read_json(pd.io.common.StringIO(result["model"]))
    assert np.isfinite(model_df["Coef"].to_numpy()).all()


def test_max_held_event_fraction_aborts_when_guard_holds_too_much() -> None:
    # An extreme risk_set_min_change on a small dataset forces almost every
    # release to be held; a tight max_held_event_fraction must then abort
    # with a clear error instead of silently returning a degraded fit.
    df = _survival_df(30)
    result = _run_central(
        [df], time_col="time", outcome_col="event", expl_vars=["age"],
        risk_set_min_change=25, max_held_event_fraction=0.1,
    )
    assert "error" in result
    assert "held" in result["error"]
    assert "max_held_event_fraction" in result["error"]


def test_risk_set_min_change_validation_rejects_less_than_one() -> None:
    df = _survival_df(30)
    result = _run_central(
        [df], time_col="time", outcome_col="event", expl_vars=["age"], risk_set_min_change=0
    )
    assert "risk_set_min_change" in result["error"]


def test_max_held_event_fraction_validation_rejects_out_of_range() -> None:
    df = _survival_df(30)
    result = _run_central(
        [df], time_col="time", outcome_col="event", expl_vars=["age"], max_held_event_fraction=1.5
    )
    assert "max_held_event_fraction" in result["error"]
