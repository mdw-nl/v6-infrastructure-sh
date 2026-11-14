import numpy as np
import pandas as pd
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="logistic_regression",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _separable_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x > 0).astype(int)
    return pd.DataFrame({"x": x, "x2": x2, "y": y})


def _base_kwargs(**overrides) -> dict:
    kwargs = {
        "feature_cols": ["x", "x2"], "target_col": "y", "n_rounds": 3,
        "local_epochs": 2, "batch_ratio": 0.3, "seed": 42,
    }
    kwargs.update(overrides)
    return kwargs


def test_central_is_reproducible_given_the_same_seed() -> None:
    # Two otherwise-identical runs over the same data with the same seed
    # must produce bit-identical trained weights -- previously each call to
    # partial()'s unseeded batch sample made every run (and every round)
    # non-reproducible even with a fixed seed.
    df = _separable_df(100, seed=1)
    result_a = _run_central([df], **_base_kwargs())
    result_b = _run_central([df], **_base_kwargs())
    assert result_a["state_dict"] == result_b["state_dict"]


def test_different_seeds_produce_different_weights() -> None:
    # A sanity check that the reproducibility above isn't just because
    # training is a no-op regardless of the batch: a different seed must
    # still change the result.
    df = _separable_df(100, seed=1)
    result_a = _run_central([df], **_base_kwargs(seed=42))
    result_b = _run_central([df], **_base_kwargs(seed=7))
    assert result_a["state_dict"] != result_b["state_dict"]


def test_batches_differ_across_rounds_but_not_across_repeats_of_the_same_round() -> None:
    # partial() must not replay the exact same batch every round for a
    # fixed seed -- that would starve later rounds of new training signal
    # -- but repeating the *same* round_num with the same seed must be
    # fully reproducible. Non-zero starting weights make the loss sensitive
    # to exactly which rows ended up in the batch (with all-zero weights
    # every row predicts 0.5, and the loss collapses to a constant
    # regardless of which rows were sampled).
    df = _separable_df(100, seed=1)

    def _loss_for_round(round_num: int) -> float:
        client = MockAlgorithmClient(datasets=[[{"database": df, "input_data": {}}]], module="logistic_regression")
        org_ids = [organization["id"] for organization in client.organization.list()]
        task = client.task.create(
            input_={
                "method": "partial",
                "kwargs": {
                    "feature_cols": ["x", "x2"], "target_col": "y",
                    "state_dict": {"linear.weight": [[2.0, -1.5]], "linear.bias": [0.3]},
                    "global_mean": [0.0, 0.0], "global_std": [1.0, 1.0],
                    "train_ratio": 1.0, "batch_ratio": 0.3, "seed": 42,
                    "local_epochs": 1, "round_num": round_num,
                },
            },
            organizations=[org_ids[0]],
        )
        result = client.wait_for_results(task.get("id"))[0]
        return result["loss"]

    assert _loss_for_round(1) == _loss_for_round(1)
    assert _loss_for_round(1) != _loss_for_round(2)
