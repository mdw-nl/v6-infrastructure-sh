---
name: v6-infrastructure-skill
description: Use this skill when setting up or running vantage6 infrastructure tests for any algorithm repo, both locally and in GitHub Actions, with pinned infra SHA and repo-specific test configs/data.
---

# V6 Infrastructure Skill

Use this skill to run reusable vantage6 infra tests for any algorithm package without copying infra code into each repo.

## When to use

- You need to run infra-backed tests for an algorithm repo.
- The algorithm repo does not vendor `v6-infrastructure-sh`.
- You want reproducible CI with infra pinned to a commit SHA.

## Required inputs

Keep these files in the algorithm repo:

- `tests/infra/config.env`
- `tests/infra/nodes.env`
- `tests/data/*`
- `tests/infra/run_algo_smoke.sh`
- `tests/infra/vars.env.example` (recommended placeholders for local env vars)

`nodes.env` format:

```text
name|api_key|db_uri|db_type|db_label
```

Recommended placeholders for humans and LLM agents:

```bash
export PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}"
export REGISTRY_PORT="${REGISTRY_PORT:-5001}"  # macOS often prefers 5001 or 50000
export UI_ENABLED="${UI_ENABLED:-false}"
```

## Local workflow

1. Clone infra harness and pin SHA.
2. Copy algorithm test configs into infra folder.
3. Run lifecycle commands with consistent env flags for `up`, `test`, and `down`.

```bash
# from algorithm repo root
ALG_ROOT="$(pwd)"
INFRA_DIR="$ALG_ROOT/tools/v6-infra"
git clone https://github.com/mdw-nl/v6-infrastructure-sh.git "$INFRA_DIR"
cd "$INFRA_DIR"
git checkout <INFRA_SHA>

cp "$ALG_ROOT/tests/infra/config.env" infrastructure/config.env
cp "$ALG_ROOT/tests/infra/nodes.env" infrastructure/nodes.env

cd infrastructure
PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}" ./infra.sh preflight
PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}" ENVIRONMENT=CI UI_ENABLED="${UI_ENABLED:-false}" ./infra.sh up
bash "$ALG_ROOT/tests/infra/run_algo_smoke.sh"
PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}" UI_ENABLED="${UI_ENABLED:-false}" ./infra.sh test
PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}" ENVIRONMENT=CI UI_ENABLED="${UI_ENABLED:-false}" ./infra.sh down
```

If permissions are stale or task creation fails unexpectedly, do a clean restart:

```bash
./infra.sh down
rm -rf "$HOME/.local/share/vantage6/server/demoserver"
rm -rf "$HOME/.local/share/vantage6/node/"*
PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}" ENVIRONMENT=CI UI_ENABLED="${UI_ENABLED:-false}" ./infra.sh up
```

## Rolesets in `entities.yaml`

When using this harness-generated entities, explicit user roles are not set in
the YAML. On vantage6 `4.13.3` the import flow assigns organization-scoped
`super` roles to imported org users by default.

If users can authenticate but receive `You lack the permission to do that!`
when creating tasks, first suspect stale server state and recreate infra
(`infra.sh down` + clean local state + `infra.sh up`) before changing role
assumptions.

## Local registry for custom images

If tasks stay in `non-existing Docker image`, publish the image to a reachable
local registry and reference that image in task payloads:

```bash
REGISTRY_PORT="${REGISTRY_PORT:-5001}"  # macOS fallback: 50000
docker run -d --restart unless-stopped -p "${REGISTRY_PORT}:5000" --name v6-local-registry registry:2
docker tag <local-image>:<tag> localhost:${REGISTRY_PORT}/<local-image>:<tag>
docker push localhost:${REGISTRY_PORT}/<local-image>:<tag>
```

Then use `localhost:${REGISTRY_PORT}/<local-image>:<tag>` as the task `image`.

## Client API compatibility (vantage6 4.13.x)

When collecting results in smoke scripts:

- Use `client.result.from_task(task_id=<id>)` (not `client.result.list(...)`).
- Use `client.run.list(task=<id>)` (argument name is `task`, not `task_id`).

For portable scripts, avoid hardcoded host-specific paths; prefer env vars and placeholders.

## Dependency guardrails for algorithm repos

- Avoid `vantage6-tools` in requirements; use `vantage6-client` and `vantage6-algorithm-tools`.
- Keep smoke scripts non-interactive (`MPLBACKEND=Agg` if plots are produced).

## GitHub Actions workflow pattern

Use `actions/checkout` twice: once for the algorithm repo and once for infra harness at pinned SHA.

```yaml
- uses: actions/checkout@v4

- name: Checkout infra harness
  uses: actions/checkout@v4
  with:
    repository: mdw-nl/v6-infrastructure-sh
    ref: <INFRA_SHA>
    path: tools/v6-infra

- name: Inject repo test config
  run: |
    cp tests/infra/config.env tools/v6-infra/infrastructure/config.env
    cp tests/infra/nodes.env tools/v6-infra/infrastructure/nodes.env

- name: Start infra
  run: PYTHON_INTERPRETER=python3.12 ENVIRONMENT=CI UI_ENABLED=false tools/v6-infra/infrastructure/infra.sh up

- name: Algorithm smoke
  run: bash tests/infra/run_algo_smoke.sh

- name: Infra smoke
  run: PYTHON_INTERPRETER=python3.12 UI_ENABLED=false tools/v6-infra/infrastructure/infra.sh test

- name: Shutdown infra
  if: always()
  run: PYTHON_INTERPRETER=python3.12 ENVIRONMENT=CI UI_ENABLED=false tools/v6-infra/infrastructure/infra.sh down
```

## Guardrails

- Do not duplicate infra scripts into algorithm repos.
- Always pin infra by commit SHA in CI.
- Keep algorithm-specific data and expectations in the algorithm repo only.
- Keep smoke tests deterministic and non-interactive.
- If Docker daemon is unavailable, fail fast at `preflight`.
- Prefer path-agnostic env vars/placeholders over machine-specific absolute paths.
- Use the same infra env flags for `up`, `test`, and `down` to avoid false negatives.

## Failure triage

1. `preflight` fails: Docker/runtime or missing config paths.
2. `up` fails: server/node startup config mismatch.
3. `run_algo_smoke.sh` fails: algorithm/package/runtime issue.
4. `infra.sh test` fails: verify `UI_ENABLED`/`NODES_CONFIG` match how infra was started.
5. Task status `non-existing Docker image`: use local registry with configurable port and retag image.
6. `down` fails: teardown residue; rerun and inspect remaining containers.
