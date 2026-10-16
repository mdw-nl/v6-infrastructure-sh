# Argos CNN (federated CT/GTV segmentation)

`algorithms/argos_cnn/` is a PyTorch port of `mod_resnet` from the original `argosfeddeep`
(TensorFlow) repo: a ResNet-encoder / PSP-style multi-scale-fusion decoder that segments lung
tumors (GTV) on 2.5D (3-slice) CT stacks. It needs a per-slice NIfTI manifest, not the default
LUNG1 CSVs — see **Data contract** below, and the module docstring in `argos_cnn.py` for the
exact, authoritative detail (this file is a convenience copy for anyone who only has the
published image, not the source).

## Running it

1. **Generate the node data** (synthetic phantom CT/GTV NIfTI files + manifests — see
   `generate_argos_data.py`'s docstring for exactly what's realistic vs. fake about it):

   ```bash
   uv run python data/generate_argos_data.py
   ```

   Writes `data/argos/{alpha,beta,gamma,theta}.csv` and `data/argos/nifti/<org>/...`.

2. **Start the network with that data** — uses `infrastructure/nodes.argos.env`, which also
   mounts each org's `data/argos/nifti/<org>` folder read-only at `/mnt/nifti` (see the root
   README's folder-database note):

   ```bash
   cd infrastructure
   ENVIRONMENT=DEV ./infra.sh --nodes argos up
   ```

3. **Build the algorithm image** (from repo root — copies both `argos_cnn.py` and `model.py`):

   ```bash
   docker build -t argos_cnn:latest algorithms/argos_cnn/
   ```

   Or skip building entirely and point `run_study.py`'s `ALGORITHM_IMAGE` at
   `ghcr.io/mdw-nl/v6-infrastructure-sh/argos_cnn:latest` instead — see the root README's
   **Using a published image instead of building**.

4. **Run it on the network**:

   ```bash
   uv run python algorithms/argos_cnn/run_study.py
   ```

   Submits federated training (FedAvg, weighted by each node's dataset size), waits for all
   rounds, and writes the trained weights to `argos_cnn_trained_weights.pt`.

## Data contract

This algorithm assumes CT data: axial slices, already consistently oriented and resampled to
the same voxel spacing across every patient/node, with `gt_path` label values in
`[0, NUM_CLASSES)`. It does no orientation/spacing validation (the code never reads a NIfTI's
affine) and no modality-specific normalization beyond CT's HU windowing — see `argos_cnn.py`'s
module docstring for the full data contract.

Within that contract, these are all real `central()`/`partial()` overrides, not fixed to this
repo's lung-tumor demo:

- `num_classes` — segmentation target, e.g. binary tumor vs. multi-organ.
- `num_channels` — 2.5D neighbor-slice context width.
- `min_bound`/`max_bound` — HU windowing preset (e.g. lung window vs. mediastinal window).
- `random_seed` — both the initial model weights and each round's batch sampling.
- `patient_col`/`slice_col`/`ct_col`/`gt_col`/`label_col` — manifest column names, if your
  manifest uses different names than this repo's demo data.

A manifest missing any required column raises a clear `ValueError` instead of silently
contributing nothing, while a node with a correctly-shaped but genuinely empty dataset still
contributes 0 without raising.

## Notes

- Unlike the other algorithms, hyperparameters (`N_ROUNDS`/`LOCAL_STEPS`/`BATCH_SIZE`/
  `LEARNING_RATE`/`PATCH_SIZE`/`NUM_CLASSES`/`NUM_CHANNELS`/`MIN_BOUND`/`MAX_BOUND`/
  `RANDOM_SEED`, plus the manifest column names) live only in `argos_cnn.py` — `run_study.py`
  intentionally passes none, so there's a single source of truth. Edit them there and rebuild
  the image. `PATCH_SIZE` must match the manifest's actual slice resolution (and stay divisible
  by 32, a constraint of the encoder/decoder's downsampling depth) — a mismatch now raises a
  clear `ValueError` instead of a cryptic numpy broadcast error.
- Trained weights ride inside the normal vantage6 task payload as a gzip+base64 string (tens of
  MB for this model) rather than a separate file-transfer channel — this won't scale
  indefinitely to much larger models; see `argos_cnn.py`'s module docstring and vantage6's
  blob-storage mechanism (`vantage6/common/client/blob_storage.py`) as the eventual fix.
- `data/argos/` is synthetic phantom data for exercising the pipeline end to end (I/O,
  normalization, FedAvg), not real anatomy — don't draw conclusions about model quality from it.
