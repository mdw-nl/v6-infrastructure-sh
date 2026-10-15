import numpy as np
import pandas as pd

from generate_beach_data import _compute_two_year_survival


def test_two_year_survival_thresholds_match_the_documented_rules() -> None:
    threshold = 2 * 365.24
    vital_status = pd.Series(
        ["dead", "dead", "alive", "alive", "unknown", "dead"],
        name="vital_status",
    )
    days = pd.Series(
        [threshold - 10, threshold + 10, threshold + 10, threshold - 10, threshold + 10, None],
        name="days",
    )

    result = _compute_two_year_survival(vital_status, days)

    assert result.iloc[0] == 0  # dead, within 2y -> 0
    assert result.iloc[1] == 1  # dead, past 2y -> 1
    assert result.iloc[2] == 1  # alive, past 2y -> 1
    assert np.isnan(result.iloc[3])  # alive, insufficient follow-up -> NaN
    assert np.isnan(result.iloc[4])  # unrecognized vital_status -> NaN
    assert np.isnan(result.iloc[5])  # dead but days unknown -> NaN (can't place vs. threshold)


def test_two_year_survival_is_not_just_vital_status() -> None:
    # Regression test for the actual bug: a naive implementation collapses to
    # `out[dead] = 0; out[alive] = 1` regardless of `days`, i.e. it computes
    # plain vital status. This dataset has a dead patient who survived past
    # two years (should be 1, not 0) and an alive patient without enough
    # follow-up (should be NaN, not 1) -- both would fail under that logic.
    vital_status = pd.Series(["dead", "alive"])
    days = pd.Series([2 * 365.24 + 100, 30])

    result = _compute_two_year_survival(vital_status, days)

    assert result.iloc[0] == 1
    assert np.isnan(result.iloc[1])
