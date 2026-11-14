import numpy as np
import pandas as pd
from kaplan_meier import apply_risk_set_merge_guard
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="kaplan_meier",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


# ── apply_risk_set_merge_guard (pure function) ───────────────────────────────


def test_default_min_change_of_one_never_holds() -> None:
    n_risk = np.array([10, 9, 7, 4])
    n_events = np.array([1, 2, 3, 4])
    out_risk, out_events, held = apply_risk_set_merge_guard(n_risk, n_events, min_change=1)
    assert not held.any()
    np.testing.assert_array_equal(out_risk, n_risk)
    np.testing.assert_array_equal(out_events, n_events)


def test_single_departure_is_held_and_its_events_fold_forward() -> None:
    # 10 -> 9 at-risk is exactly one departure; min_change=2 must hold step 1
    # at step 0's n_risk, and fold its 2 events into step 2 (which clears
    # the threshold on its own: 10 -> 7 is 3 departures since the last
    # release, so step 2 releases fresh and absorbs step 1's carried events).
    n_risk = np.array([10, 9, 7, 4])
    n_events = np.array([1, 2, 3, 4])
    out_risk, out_events, held = apply_risk_set_merge_guard(n_risk, n_events, min_change=2)

    assert held.tolist() == [False, True, False, False]
    assert out_risk.tolist() == [10, 10, 7, 4]
    assert out_events.tolist() == [1, 0, 5, 4]  # step1's 2 events folded into step2: 3 + 2
    assert out_events.sum() == n_events.sum()


def test_total_events_are_conserved_regardless_of_holding_pattern() -> None:
    rng = np.random.default_rng(0)
    n_risk = np.sort(rng.integers(1, 100, size=20))[::-1]
    n_events = rng.integers(0, 5, size=20)
    for min_change in (1, 2, 3, 5, 10, 50):
        _, out_events, _ = apply_risk_set_merge_guard(n_risk, n_events, min_change)
        assert out_events.sum() == n_events.sum()


def test_held_periods_accumulate_against_the_last_true_release() -> None:
    n_risk = np.array([10, 9, 8, 7])
    n_events = np.array([1, 1, 1, 1])
    out_risk, out_events, held = apply_risk_set_merge_guard(n_risk, n_events, min_change=2)

    # step1: 10->9, 1 departed since release@0 -> held
    # step2: 10->8, 2 departed since release@0 -> released fresh
    # step3: 8->7, 1 departed since release@2 -> held
    assert held.tolist() == [False, True, False, True]
    assert out_risk.tolist() == [10, 10, 8, 8]
    assert out_events.sum() == 4


def test_zero_time_steps_is_a_no_op() -> None:
    empty = np.zeros(0, dtype=int)
    _, _, held = apply_risk_set_merge_guard(empty, empty, 5)
    assert len(held) == 0


def test_pending_events_at_the_end_fold_into_the_last_step() -> None:
    # Every step after the first is held (never enough departures) -- the
    # events from every held step must still all end up somewhere.
    n_risk = np.array([10, 10, 10, 10])
    n_events = np.array([1, 1, 1, 1])
    _, out_events, held = apply_risk_set_merge_guard(n_risk, n_events, min_change=5)
    assert held.tolist() == [False, True, True, True]
    assert out_events.sum() == 4
    assert out_events.tolist() == [1, 0, 0, 3]  # step0's own event stays; steps 1-3's fold into the last


# ── end-to-end via central() ─────────────────────────────────────────────────


def _survival_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Survival.time": rng.integers(1, 200, size=n).astype(float),
            "deadstatus.event": rng.integers(0, 2, size=n),
        }
    )


def test_strict_risk_set_min_change_preserves_total_event_count() -> None:
    df = _survival_df(60, seed=1)
    baseline = _run_central([df], time_col="Survival.time", event_col="deadstatus.event", step_days=5)
    guarded = _run_central(
        [df], time_col="Survival.time", event_col="deadstatus.event", step_days=5, risk_set_min_change=5,
    )
    assert "error" not in guarded
    assert guarded["n_events"] == baseline["n_events"]


def test_max_held_event_fraction_aborts_when_guard_holds_too_much() -> None:
    df = _survival_df(30, seed=2)
    result = _run_central(
        [df], time_col="Survival.time", event_col="deadstatus.event", step_days=1,
        risk_set_min_change=25, max_held_event_fraction=0.1,
    )
    assert "error" in result
    assert "held" in result["error"] or "folded" in result["error"]
    assert "max_held_event_fraction" in result["error"]


def test_risk_set_min_change_validation_rejects_less_than_one() -> None:
    df = _survival_df(30, seed=3)
    result = _run_central(
        [df], time_col="Survival.time", event_col="deadstatus.event", risk_set_min_change=0
    )
    assert "risk_set_min_change" in result["error"]


def test_max_held_event_fraction_validation_rejects_out_of_range() -> None:
    df = _survival_df(30, seed=4)
    result = _run_central(
        [df], time_col="Survival.time", event_col="deadstatus.event", max_held_event_fraction=1.5
    )
    assert "max_held_event_fraction" in result["error"]
