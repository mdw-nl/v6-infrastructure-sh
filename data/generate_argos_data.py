#!/usr/bin/env python3
"""Generate synthetic test data for the argos_cnn (federated ModResNet CT/GTV
segmentation) algorithm.

`data/argos/` is gitignored (like `data/beach/`) because it's mostly binary
NIfTI files — not something that belongs in git history. This script
regenerates it deterministically after a fresh clone, the same way
`generate_beach_data.py` regenerates `data/beach/`.

Everything produced here is a fake phantom, not real patient data: each "CT"
slice is a procedurally generated torso/lung outline (body ellipse, two lung
fields, HU-like intensity ranges) with a synthetic nodule placed at a fixed
per-patient location on slices marked `has_tumor=1`; the matching GT file
marks exactly that nodule. This exists purely to exercise the pipeline
(sampling, 2.5D stacking, normalization, training, FedAvg) end to end — it
has no relationship to any real dataset and should never be used to draw
conclusions about model performance.

Output layout, per node/org (matches argos_cnn.py's REQUIRED_COLS and the
nodes.argos.env / node_extra_mounts wiring in infrastructure/functions.sh):

    data/argos/<org>.csv                                  patient_id,slice_index,ct_path,gt_path,has_tumor
    data/argos/nifti/<org>/<patient_id>/<slice>_ct.nii.gz  2D (512x512) synthetic CT slice
    data/argos/nifti/<org>/<patient_id>/<slice>_gt.nii.gz  2D (512x512) binary tumor mask

`ct_path`/`gt_path` in the CSVs are written as container-side paths
(`/mnt/nifti/...`), matching where the node mounts each org's
`data/argos/nifti/<org>` folder read-only inside algorithm containers — not
host paths. See argos_cnn/argos_cnn.py's module docstring for why.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

# Matches this repo's actual node/org names (see infrastructure/nodes.argos.env).
ORG_SEEDS = {"alpha": 42, "beta": 43, "gamma": 44, "theta": 45}

CSV_COLUMNS = ["patient_id", "slice_index", "ct_path", "gt_path", "has_tumor"]
AFFINE = np.eye(4)


def _phantom_masks(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Body outline + two lung fields, as boolean masks, for a `size x size` slice."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2
    scale = size / 512.0

    body = ((xx - cx) / (220 * scale)) ** 2 + ((yy - cy) / (240 * scale)) ** 2 <= 1
    left_lung = ((xx - (cx - 110 * scale)) / (85 * scale)) ** 2 + ((yy - cy) / (150 * scale)) ** 2 <= 1
    right_lung = ((xx - (cx + 110 * scale)) / (85 * scale)) ** 2 + ((yy - cy) / (150 * scale)) ** 2 <= 1
    lungs = (left_lung | right_lung) & body
    return body, lungs


def _make_ct_slice(
    rng: np.random.Generator,
    size: int,
    body: np.ndarray,
    lungs: np.ndarray,
    tumor_center: tuple[int, int] | None,
    tumor_radius: int,
) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    ct = np.full((size, size), -1000, dtype=np.float32)  # air outside the body
    ct[body] = 40 + rng.normal(0, 15, size=int(body.sum()))  # soft tissue
    ct[lungs] = -750 + rng.normal(0, 40, size=int(lungs.sum()))  # lung parenchyma

    if tumor_center is not None:
        ty, tx = tumor_center
        nodule = ((xx - tx) ** 2 + (yy - ty) ** 2) <= tumor_radius**2
        nodule &= lungs
        if nodule.sum() > 0:
            ct[nodule] = 30 + rng.normal(0, 10, size=int(nodule.sum()))
    return ct.astype(np.int16)


def _make_gt_slice(
    size: int,
    lungs: np.ndarray,
    tumor_center: tuple[int, int] | None,
    tumor_radius: int,
) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    gt = np.zeros((size, size), dtype=np.uint8)
    if tumor_center is not None:
        ty, tx = tumor_center
        nodule = ((xx - tx) ** 2 + (yy - ty) ** 2) <= tumor_radius**2
        nodule &= lungs
        gt[nodule] = 1
    return gt


def _generate_org(
    org: str,
    seed: int,
    output_dir: Path,
    num_patients: int,
    num_slices: int,
    patch_size: int,
) -> int:
    body, lungs = _phantom_masks(patch_size)
    rng = np.random.default_rng(seed)

    nifti_dir = output_dir / "nifti" / org
    rows: list[dict] = []

    for p in range(1, num_patients + 1):
        patient_id = f"LUNG1-{org.upper()}-{p:03d}"
        patient_dir = nifti_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        # One tumor per patient, in a random lung, sized/placed once and reused
        # across that patient's tumor-positive slices (a real GTV doesn't
        # relocate slice to slice).
        side = rng.choice([-1, 1])
        cy = patch_size // 2
        cx = patch_size // 2
        tumor_center = (
            cy + int(rng.integers(-60, 60) * patch_size / 512),
            cx + side * int(110 * patch_size / 512) + int(rng.integers(-30, 30) * patch_size / 512),
        )
        tumor_radius = int(rng.integers(12, 22) * patch_size / 512)

        tumor_len = int(rng.integers(3, 6))
        tumor_start = int(rng.integers(2, max(3, num_slices - tumor_len - 2)))
        tumor_slices = set(range(tumor_start, tumor_start + tumor_len))

        for s in range(num_slices):
            has_tumor = s in tumor_slices
            slice_center = tumor_center if has_tumor else None
            slice_radius = tumor_radius if has_tumor else 0

            ct_slice = _make_ct_slice(rng, patch_size, body, lungs, slice_center, slice_radius)
            gt_slice = _make_gt_slice(patch_size, lungs, slice_center, slice_radius)

            ct_path = patient_dir / f"{s}_ct.nii.gz"
            gt_path = patient_dir / f"{s}_gt.nii.gz"
            nib.save(nib.Nifti1Image(ct_slice, AFFINE), ct_path)
            nib.save(nib.Nifti1Image(gt_slice, AFFINE), gt_path)

            rows.append(
                {
                    "patient_id": patient_id,
                    "slice_index": s,
                    # Container-side path: this org's nifti/ folder is mounted
                    # read-only at /mnt/nifti inside algorithm containers.
                    "ct_path": f"/mnt/nifti/{patient_id}/{s}_ct.nii.gz",
                    "gt_path": f"/mnt/nifti/{patient_id}/{s}_gt.nii.gz",
                    "has_tumor": int(has_tumor),
                }
            )

    csv_path = output_dir / f"{org}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output = repo_root / "data" / "argos"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--num-patients", type=int, default=5, help="Patients per org")
    parser.add_argument("--num-slices", type=int, default=20, help="Axial slices per patient")
    parser.add_argument("--patch-size", type=int, default=512, help="Slice height/width in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_patients < 1:
        raise SystemExit("--num-patients must be at least 1")
    if args.num_slices < 3:
        raise SystemExit("--num-slices must be at least 3 (need room for 2.5D neighbor slices)")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for org, seed in ORG_SEEDS.items():
        n_rows = _generate_org(
            org=org,
            seed=seed,
            output_dir=output_dir,
            num_patients=args.num_patients,
            num_slices=args.num_slices,
            patch_size=args.patch_size,
        )
        total_rows += n_rows
        print(f"{org}: wrote {output_dir / f'{org}.csv'} ({n_rows} rows) and {n_rows * 2} NIfTI files")

    print(f"Done. {total_rows} manifest rows, {total_rows * 2} NIfTI files under {output_dir / 'nifti'}")


if __name__ == "__main__":
    main()
