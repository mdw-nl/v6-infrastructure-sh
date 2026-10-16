# Vantage6 Local Infrastructure Harness

This repository provides a reusable, config-driven local vantage6 infrastructure for any algorithm package and data layout.

The default runtime now assumes GitHub Container Registry images:

- `ghcr.io/mdw-nl/vantage6/infrastructure/server-lite`
- `ghcr.io/mdw-nl/vantage6/infrastructure/node-lite`
- `ghcr.io/mdw-nl/vantage6/infrastructure/ui`

Harbor is intentionally no longer part of the default path.

## What changed

The infrastructure is now driven by:

- `infrastructure/config.env`: runtime defaults (Python version, v6 version, server/UI settings, paths)
- `infrastructure/nodes.<profile>.env`: node specs (`name|api_key|db_uri|db_type|db_label`) — `nodes.lung1.env` is the default profile, used when no `--nodes` flag is given (see "Switching node profiles" below)
- generated runtime artifacts in `infrastructure/generated/`

No hardcoded `alpha/beta/gamma` logic is required anymore. Any number of nodes can be used.

## Quick start

1. Edit `infrastructure/config.env` and `infrastructure/nodes.lung1.env`.
2. Run preflight checks:

```bash
cd infrastructure
./infra.sh preflight
```

3. Start infrastructure:

```bash
cd infrastructure
ENVIRONMENT=DEV ./infra.sh up
```

4. Run smoke tests:

```bash
cd infrastructure
./infra.sh test
```

5. Tear down:

```bash
cd infrastructure
./infra.sh down
```

## CI compatibility

Legacy entrypoints remain and map to the same flow:

- `infrastructure/setup.sh`
- `infrastructure/shutdown.sh`

The authoritative smoke environment is `ubuntu-latest` or another amd64 host. ARM developer machines are supported on a best-effort basis only.

If the published GHCR `server-lite` / `node-lite` / `ui` images are only available locally as amd64 images, `infra.sh up` now first installs `qemu-x86_64` binfmt when needed, then retries them with `DOCKER_DEFAULT_PLATFORM=linux/amd64`. Set `V6_AUTO_INSTALL_BINFMT=false` if you want to manage emulation yourself. If Docker still cannot execute the image through emulation, the harness fails fast with a clear architecture probe error instead of starting a partial stack and failing later during entity import.

Cleanup removes stale local Vantage6 state and Docker networks for the configured
server and node names before startup. This prevents old tasks, databases, and
networks from contaminating the next run. It does not clear a live uWSGI/qemu
socket backlog inside a running server process; if the server stops responding
under emulation, stop the stack and retry on amd64 for an authoritative result.

## Node spec examples

`infrastructure/nodes.<profile>.env` supports mixed backends when `RUN_CONTEXT_FILE=false`:

```text
alpha|<api_key>|../data/alpha.csv|csv|default
beta|<api_key>|postgresql://user:pass@db:5432/demo|sql|warehouse
```

When `RUN_CONTEXT_FILE=true` (the default for standalone STRATA-FIT algorithms),
the harness requires each node database to be a local file-backed CSV entry:

```text
gamma|<api_key>|/absolute/path/to/data_bucket3.csv|csv|default
```

Do not use `sql`, `postgresql://...`, or other URI-backed sources with
`RUN_CONTEXT_FILE=true`; the vantage6 node cannot materialize those as
`RUN_CONTEXT_FILE` inputs and will fail before the algorithm starts.

If `db_uri` is empty, it defaults to `${DATA_DIR_DEFAULT}/<name>.csv`.

An optional second, read-only **folder** database can be added via 3 extra columns — `name|api_key|db_uri|db_type|db_label|extra_db_uri|extra_db_type|extra_db_label` — bind-mounted read-only at `/mnt/<extra_db_label>` inside algorithm containers. `argos_cnn` uses this to mount each org's NIfTI folder alongside its CSV manifest; see `infrastructure/nodes.argos.env`.

### Switching node profiles

Every algorithm's data layout — including the default LUNG1 CSVs — lives in its own `infrastructure/nodes.<profile>.env` file (`nodes.lung1.env`, `nodes.beach.env`, `nodes.argos.env`, ...). There is no bare `nodes.env`; the default is just the `lung1` profile like any other. Point any command at a profile with `--nodes <profile>` (before or after the command name):

```bash
./infra.sh --nodes lung1 up    # same as plain `./infra.sh up`
./infra.sh --nodes beach up
./infra.sh preflight --nodes argos
```

With no `--nodes` flag, `infra.sh` falls back to the `lung1` profile (`config.env`'s `NODES_CONFIG` default). `./infra.sh help` lists every profile it discovered. Adding a new algorithm's data layout only requires dropping in a new `nodes.<profile>.env` file — no changes to `infra.sh` itself.

Generated node YAML keeps the runtime-critical fields explicit:

```yaml
databases:
  - label: default
    type: csv
    uri: /absolute/path/to/data.csv
    mount_mode: ro
images:
  node: ghcr.io/mdw-nl/vantage6/infrastructure/node-lite:4.14.0-rc8
share_config: false
share_algorithm_logs: false
run_context_file: true
prometheus:
  enabled: false
```

For operator-facing configs, prefer digest-pinned image refs.

## Entities and roles

`infra.sh up` always generates an `entities.generated.yaml` and uploads it into
the server container (`vserver-local import ...`).

The generated entities currently do not set explicit user roles. On
vantage6 `4.13.3` in this harness, imported org users receive an
organization-scoped `super` role by default (verified from the server DB).

If you see permission errors on task creation (`You lack the permission to do that!`),
it is usually stale local server state. Run `infra.sh down`, clear local server DB
state, and run `infra.sh up` again.

## Local image registry

This is for a throwaway local registry, useful when debugging a node that can't see a
locally-built image (e.g. Docker context isolation). If you just want to skip building
an `algorithms/` image altogether, use the published GHCR image instead — see
[algorithms/README.md](algorithms/README.md)'s **Using a published image instead of building**.

If nodes report `non-existing Docker image`, use a local registry and submit
tasks with a registry-backed image reference:

```bash
docker run -d --restart unless-stopped -p 5000:5000 --name v6-local-registry registry:2
docker tag local/v6-sklearn-linear-py:dev localhost:5000/v6-sklearn-linear-py:dev
docker push localhost:5000/v6-sklearn-linear-py:dev
```

Then use `localhost:5000/v6-sklearn-linear-py:dev` in task creation.

## Data

The `data/` directory holds the default LUNG1 CSVs plus everything that generates or checks per-node data for the non-default node profiles. Full details (including a disambiguation of three unrelated "datavalgen" things that live here) are in [data/README.md](data/README.md) — summary:

| Path | What it is |
|---|---|
| `data/lung1/` | Default LUNG1 CSVs, used by `nodes.lung1.env` (the default profile). Static, hand-crafted synthetic data (not a real patient dataset) — see [data/lung1/README.md](data/lung1/README.md) for exact columns, row counts, and per-algorithm default usage |
| `data/generate_beach_data.py` | Generates BEACH-schema CSVs (an alternative dataset for `logistic_regression`/`admm_logistic_regression`), used with `nodes.beach.env` |
| `data/generate_argos_data.py` | Generates synthetic CT/GTV NIfTI + manifest data for `argos_cnn`, used with `nodes.argos.env` — no datavalgen involvement, since this data is NIfTI images, not a flat schema-able CSV |
| `data/datavalgen/` | **Unrelated to the three datasets above.** A self-contained demo sub-project (its own `pyproject.toml`/`.venv`) with its own Pydantic schema ("example" model) + CLI for generating/validating fictional patient CSVs by hand, plus a vantage6 algorithm (`v6-validate/`) that runs that same validation federated across nodes. All setup, Docker, and run instructions live in [data/datavalgen/README.md](data/datavalgen/README.md) — see "Federated datavalgen validate" below for how it fits into this harness |

Each data source's matching node profile (`infrastructure/nodes.<profile>.env`) and the algorithm that consumes it are documented together in the algorithm-specific sections below.

## Algorithms

The `algorithms/` directory contains federated learning algorithms that run on top of the vantage6 infrastructure. Each algorithm has its own folder with:

- An algorithm module (the code that runs on each node and the central orchestrator)
- A `Dockerfile` to build the node image
- A `run_study.py` script to submit the task from your machine

Available algorithms:

| Folder | Description |
|---|---|
| `average/` | Federated average of a single column |
| `logistic_regression/` | Federated logistic regression with normalization, batch training, and per-node evaluation |
| `kaplan_meier/` | Federated Kaplan-Meier survival curve with 95% CI, a matplotlib plot, and optional Gaussian/Poisson noise injection on event times for privacy |
| `coxph/` | Federated Cox proportional hazards regression (hazard ratios across covariates) via federated Newton-Raphson iterations |
| `fed_statistics/` | Federated descriptive statistics (counts, binned counts, min/max, mean, std, bootstrap quantiles, nrows, nans) with threshold + secondary suppression for privacy |
| `argos_cnn/` | Federated ModResNet (PyTorch) for CT/GTV tumor segmentation — see **Argos CNN** below |
| `admm_logistic_regression/` | Federated logistic regression via ADMM consensus — reproduces a centralized/pooled fit's coefficients, on the same default LUNG1 data as `logistic_regression/` — see `algorithms/admm_logistic_regression/README.md` |

Each algorithm file has a block of **user-configurable variables** near the top (feature columns, target column, learning rate, number of rounds, train/test ratio, etc.). These act only as fallback defaults — see [algorithms/README.md](algorithms/README.md)'s **Changing variables without rebuilding the image** for how to override the important ones (the data columns) per run from `run_study.py`.

Note there are two logistic regression implementations in this repo, both taking the same `feature_cols`/`target_col` arguments and assuming already-numeric features plus an already-binary target (no categorical encoding or outcome derivation — that's a data-preparation concern, not this algorithm's): `logistic_regression/` trains a PyTorch model with federated averaging of weights across rounds (an approximation whose accuracy depends on rounds/epochs/batch size), while `admm_logistic_regression/` uses ADMM consensus optimization, which converges to the same coefficients as a centralized/pooled fit. Neither needs any special node setup — both run against the default LUNG1 data out of the box — see `algorithms/admm_logistic_regression/README.md` for a full side-by-side comparison of the two, including how to point either one at the alternative BEACH dataset instead.

Building images, running studies, using the published GHCR images instead of building, overriding variables without rebuilding, the full per-algorithm data-contracts table, and adding your own algorithm are all documented in [algorithms/README.md](algorithms/README.md) — that content applies uniformly across every algorithm here, so it lives in one place instead of repeated per algorithm.

## Argos CNN (federated CT/GTV segmentation)

`algorithms/argos_cnn/` is a PyTorch port of `mod_resnet` from the original `argosfeddeep`
(TensorFlow) repo: a ResNet-encoder / PSP-style multi-scale-fusion decoder that segments lung
tumors (GTV) on 2.5D (3-slice) CT stacks. It needs a per-slice NIfTI manifest, not the default
LUNG1 CSVs — see [algorithms/argos_cnn/README.md](algorithms/argos_cnn/README.md) for the full
data contract, hyperparameters, and known limitations (trained-weights payload size, etc.).

1. Generate the node data: `uv run python data/generate_argos_data.py`
2. Start the network with that data: `cd infrastructure && ENVIRONMENT=DEV ./infra.sh --nodes argos up`
3. Build the algorithm image: `docker build -t argos_cnn:latest algorithms/argos_cnn/`
4. Run it: `uv run python algorithms/argos_cnn/run_study.py`

See [algorithms/argos_cnn/README.md](algorithms/argos_cnn/README.md) for what each step actually does.

## Federated datavalgen validate

`data/datavalgen/v6-validate/` is a vantage6 algorithm that checks every node's local CSV against a `datavalgen` pydantic model *without* centralizing any data: each node validates its own file and returns only a pass/fail verdict plus an error count — never the offending cell values. The central task then reports which organizations have correctly formatted data.

Full instructions — building both of `data/datavalgen`'s Docker images, starting the network with `nodes.datavalgen.env`, and running the task — are all in [data/datavalgen/README.md](data/datavalgen/README.md), so they stay in one place instead of drifting out of sync with this file.

## Notes

- Docker daemon must be available before running setup/test.
- `STRICT_DATA_CHECKS=true` enforces local CSV existence checks.
- UI can be disabled with `UI_ENABLED=false`.
- Optional uWSGI mitigation: build a patched local `server-lite` image that adds
  `--listen 1024`, `--need-app`, maybe `--stats :9191`, and possibly
  `--harakiri`. This can reduce queue-full noise, but it is a buffer, not the
  real fix. Avoid adding multiple uWSGI workers until tested because this
  harness uses sqlite-backed local server state.
- Version/build provenance matters here. The MDW `server-lite` images inspected
  at `4.14.0-rc7`, `4.14.0-rc8`, and `4.14.0-rc9` all package
  `vantage6-server 4.14.0` and start it with the same single uWSGI command.
  Retrying adjacent rc tags is unlikely to fix uWSGI queue overflow unless the
  underlying Vantage6 server startup/runtime changes.
- Recommended workflow is:
  1. validate the target algorithm repo in a fresh `/tmp` venv
  2. run a local container smoke for `RUN_CONTEXT_FILE`
  3. only then use this harness
