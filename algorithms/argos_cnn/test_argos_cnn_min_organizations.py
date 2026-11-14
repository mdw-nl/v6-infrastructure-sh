import nibabel as nib
import numpy as np
import pandas as pd
import torch
from argos_cnn import (
    CT_COL,
    GT_COL,
    LABEL_COL,
    PATIENT_COL,
    SLICE_COL,
)
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

PATCH_SIZE = 64


def _make_manifest(tmp_path, prefix: str, n_slices: int) -> pd.DataFrame:
    rows = []
    for i in range(n_slices):
        ct_path = tmp_path / f"{prefix}_ct_{i}.nii.gz"
        gt_path = tmp_path / f"{prefix}_gt_{i}.nii.gz"
        nib.save(
            nib.Nifti1Image(np.random.rand(PATCH_SIZE, PATCH_SIZE).astype(np.float32) * 100 - 300, np.eye(4)),
            str(ct_path),
        )
        mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.int32)
        has_tumor = i % 3 == 0
        if has_tumor:
            mask[: PATCH_SIZE // 2, : PATCH_SIZE // 2] = 1
        nib.save(nib.Nifti1Image(mask, np.eye(4)), str(gt_path))
        rows.append(
            {
                PATIENT_COL: f"{prefix}_patient",
                SLICE_COL: i,
                CT_COL: str(ct_path),
                GT_COL: str(gt_path),
                LABEL_COL: int(has_tumor),
            }
        )
    return pd.DataFrame(rows)


def _run_central(datasets: list, **kwargs) -> dict:
    client = MockAlgorithmClient(
        datasets=[[{"database": df, "input_data": {}}] for df in datasets],
        module="argos_cnn",
    )
    org_ids = [organization["id"] for organization in client.organization.list()]
    task = client.task.create(input_={"method": "central", "kwargs": kwargs}, organizations=[org_ids[0]])
    return client.wait_for_results(task.get("id"))[0]


def _base_kwargs(**overrides) -> dict:
    kwargs = {"n_rounds": 1, "local_steps": 1, "batch_size": 1, "patch_size": PATCH_SIZE}
    kwargs.update(overrides)
    return kwargs


def test_min_organizations_validation_rejects_less_than_one(tmp_path) -> None:
    torch.manual_seed(0)
    df = _make_manifest(tmp_path, "a", 10)
    result = _run_central([df], **_base_kwargs(min_organizations=0))
    assert "min_organizations" in result["error"]


def test_min_rows_validation_rejects_negative(tmp_path) -> None:
    torch.manual_seed(0)
    df = _make_manifest(tmp_path, "a", 10)
    result = _run_central([df], **_base_kwargs(min_rows=-1))
    assert "min_rows" in result["error"]


def test_min_organizations_fails_fast_when_task_has_too_few_orgs(tmp_path) -> None:
    torch.manual_seed(0)
    org0 = _make_manifest(tmp_path, "a", 10)
    org1 = _make_manifest(tmp_path, "b", 10)
    result = _run_central(
        [org0, org1], **_base_kwargs(organizations=["mock-0"], min_organizations=2)
    )
    assert "error" in result
    assert "min_organizations" in result["error"]


def test_min_rows_and_min_organizations_together_requires_qualifying_orgs(tmp_path) -> None:
    # 3 organizations are nominally included -- enough to satisfy
    # min_organizations=2 on its own -- but only one has enough manifest
    # rows to clear min_rows=8. Having enough organizations included is not
    # the same as having enough organizations with enough data.
    torch.manual_seed(0)
    org0 = _make_manifest(tmp_path, "a", 10)  # qualifies
    org1 = _make_manifest(tmp_path, "b", 2)  # excluded
    org2 = _make_manifest(tmp_path, "c", 2)  # excluded

    result = _run_central(
        [org0, org1, org2], **_base_kwargs(min_rows=8, min_organizations=2)
    )
    assert "error" in result
    assert "min_organizations" in result["error"]
    assert "min_rows" in result["error"]
    assert "mock-1" in result["error"]
    assert "mock-2" in result["error"]


def test_min_rows_excludes_low_row_organizations_but_still_succeeds(tmp_path) -> None:
    torch.manual_seed(0)
    org0 = _make_manifest(tmp_path, "a", 10)
    org1 = _make_manifest(tmp_path, "b", 2)  # excluded, min_organizations=1 so run still succeeds

    result = _run_central([org0, org1], **_base_kwargs(min_rows=8, min_organizations=1))
    assert "error" not in result
    assert result["n_rounds_completed"] == 1
    # org1's 2 rows must not count toward the round's reported total.
    assert result["round_metrics"][0]["n"] == 10


def test_model_unaffected_by_excluded_org_contaminating_fedavg(tmp_path, monkeypatch) -> None:
    # Direct check that an excluded node's weights never enter the FedAvg
    # average: with it excluded, the aggregated model must be IDENTICAL to a
    # single-node run on just the qualifying organization's data.
    import argos_cnn

    torch.manual_seed(0)
    org0 = _make_manifest(tmp_path, "a", 10)
    org1 = _make_manifest(tmp_path, "b", 2)

    monkeypatch.setattr(argos_cnn, "RANDOM_SEED", 7)
    result_with_excluded = _run_central(
        [org0, org1], **_base_kwargs(min_rows=8, min_organizations=1, seed=7)
    )

    monkeypatch.setattr(argos_cnn, "RANDOM_SEED", 7)
    result_solo = _run_central([org0], **_base_kwargs(seed=7))

    from argos_cnn import _state_dict_from_str

    sd_excluded = _state_dict_from_str(result_with_excluded["state_dict"])
    sd_solo = _state_dict_from_str(result_solo["state_dict"])
    assert torch.allclose(sd_excluded["stem_conv.weight"], sd_solo["stem_conv.weight"], atol=1e-6)
