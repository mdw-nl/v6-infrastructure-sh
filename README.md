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
- `infrastructure/nodes.env`: node specs (`name|api_key|db_uri|db_type|db_label`)
- generated runtime artifacts in `infrastructure/generated/`

No hardcoded `alpha/beta/gamma` logic is required anymore. Any number of nodes can be used.

## Quick start

1. Edit `infrastructure/config.env` and `infrastructure/nodes.env`.
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

## Node spec examples

`infrastructure/nodes.env` supports mixed backends:

```text
alpha|<api_key>|../data/alpha.csv|csv|default
beta|<api_key>|postgresql://user:pass@db:5432/demo|sql|warehouse
```

If `db_uri` is empty, it defaults to `${DATA_DIR_DEFAULT}/<name>.csv`.

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

If nodes report `non-existing Docker image`, use a local registry and submit
tasks with a registry-backed image reference:

```bash
docker run -d --restart unless-stopped -p 5000:5000 --name v6-local-registry registry:2
docker tag local/v6-sklearn-linear-py:dev localhost:5000/v6-sklearn-linear-py:dev
docker push localhost:5000/v6-sklearn-linear-py:dev
```

Then use `localhost:5000/v6-sklearn-linear-py:dev` in task creation.

## Algorithms

The `algoritms/` directory contains federated learning algorithms that run on top of the vantage6 infrastructure. Each algorithm has its own folder with:

- An algorithm module (the code that runs on each node and the central orchestrator)
- A `Dockerfile` to build the node image
- A `run_study.py` script to submit the task from your machine

Available algorithms:

| Folder | Description |
|---|---|
| `average/` | Federated average of a single column |
| `logistic_regression/` | Federated logistic regression with normalization, batch training, and per-node evaluation |
| `kaplan_meier/` | Federated Kaplan-Meier survival curve with 95% CI and a matplotlib plot |

Each algorithm file has a block of **user-configurable variables** near the top (feature columns, target column, learning rate, number of rounds, train/test ratio, etc.). These act only as fallback defaults — see **Changing variables without rebuilding the image** below for how to override the important ones (the data columns) per run from `run_study.py`.

### Python environment (uv)

A shared `pyproject.toml` in `algoritms/` covers all dependencies for running study scripts and linting algorithm code locally. Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment:

```bash
cd algoritms
uv sync
```

This creates a `.venv` inside `algoritms/`. Activate it or prefix commands with `uv run`.

### Building an algorithm image

Each algorithm must be built as a Docker image before the nodes can execute it. Build from the repo root so the tag matches what `run_study.py` expects:

```bash
# Average
docker build -t average:latest algoritms/average/

# Logistic regression
docker build -t logistic_regression:latest algoritms/logistic_regression/

# Kaplan-Meier
docker build -t kaplan_meier:latest algoritms/kaplan_meier/
```

Because the nodes run inside Docker on the same host daemon, no registry push is needed for local testing. If nodes report `non-existing Docker image` anyway, see the **Local image registry** section.

### Running a study

With the infrastructure up (`infra.sh up`) and the image built:

```bash
# from the algoritms/ directory
uv run python average/run_study.py
uv run python logistic_regression/run_study.py
uv run python kaplan_meier/run_study.py
```

The script authenticates against the local vantage6 server, submits the central task, waits for all nodes to complete, and prints the results.

### Changing variables without rebuilding the image

To test a different column/variable (e.g. `VARIABLE` in `average/run_study.py`, or `FEATURE_COLS`/`TARGET_COL` in `logistic_regression/run_study.py`), just edit the constants at the top of that algorithm's `run_study.py` and re-run it — no image rebuild needed. The constants in the algorithm file itself (`average.py`, etc.) are only the fallback defaults.

### Adding your own algorithm

1. Create a new folder under `algoritms/` with an algorithm module and a `Dockerfile`.
2. Add any new dependencies to `algoritms/pyproject.toml` and run `uv sync`.
3. Build the image: `docker build -t my-algo:latest algoritms/my-algo/`
4. Write a `run_study.py` that points at `my-algo:latest` and calls `"method": "central"`.

If the nodes report `non-existing Docker image`, see the **Local image registry** section below.

## Notes

- Docker daemon must be available before running setup/test.
- `STRICT_DATA_CHECKS=true` enforces local CSV existence checks.
- UI can be disabled with `UI_ENABLED=false`.
- Recommended workflow is:
  1. validate the target algorithm repo in a fresh `/tmp` venv
  2. run a local container smoke for `RUN_CONTEXT_FILE`
  3. only then use this harness
