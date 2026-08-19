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

Use this skill only after local package/runtime validation is already green. It is not a substitute for import-safety checks or local `run_context` smoke tests.

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
export VERSION_VANTAGE6="${VERSION_VANTAGE6:-4.14.0}"
export DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:${REGISTRY_PORT}/v6infra}"
export V6_SERVER_SOURCE_IMAGE="${V6_SERVER_SOURCE_IMAGE:-ghcr.io/mdw-nl/vantage6/infrastructure/server-lite:4.14.0-rc8}"
export V6_NODE_SOURCE_IMAGE="${V6_NODE_SOURCE_IMAGE:-ghcr.io/mdw-nl/vantage6/infrastructure/node-lite:4.14.0-rc8}"
export V6_UI_SOURCE_IMAGE="${V6_UI_SOURCE_IMAGE:-ghcr.io/mdw-nl/vantage6/infrastructure/ui:4.14.0-rc8}"
```

When writing real node configs, prefer explicit image references inside the YAML instead of relying only on shell defaults. At minimum, keep these fields visible in the node config:

```yaml
images:
  node: ghcr.io/mdw-nl/vantage6/infrastructure/node-lite@sha256:<digest>

run_context_file: true
share_config: false
share_algorithm_logs: false
prometheus:
  enabled: false
```

For production-like configs, also keep `policies.allowed_algorithms` and `databases[*].mount_mode` explicit. This makes review easier and avoids hidden runtime drift between shell env and node behavior.

Important `run_context_file` constraint:

- `run_context_file: true` only works with local file-backed node databases.
- For standalone algorithms, generated node specs should use `db_type=csv` and a local CSV path, e.g. `site-c|<api_key>|/tmp/data_bucket3.csv|csv|default`.
- Do not use `sql`, `postgresql://...`, remote URIs, or other non-file sources with `run_context_file: true`; the node will fail before the algorithm starts with a file-based database error.
- If a colleague sees `run_context_file currently supports only file-based databases`, inspect the generated node YAML first, especially `databases[0].type`, `databases[0].uri`, and whether the wrapper passed `NODES_CONFIG` to every `infra.sh` command.

## Local workflow

Recommended execution order:

1. Clean temp venv: verify package-root imports and minimal unit tests.
2. Local container smoke: verify `RUN_CONTEXT_FILE` execution without infra.
3. Infra lane: only then run `v6-infrastructure-sh`.

If step 1 fails, do not continue into infra. Recent algorithm validation failures were often caused by package-root imports pulling runtime-only code, not by the infra harness itself.

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

When debugging task failures, attach to the master org node container first and inspect the traceback there before changing infra configuration. This is the fastest way to separate container/package breakage from harness breakage.

## Registry migration note

Harbor is no longer a safe or reliable default for Vantage6 infrastructure images.

- Vantage6 publicly reported unauthorized access to the Harbor registry on April 2, 2026.
- On May 11, 2026, Vantage6 announced that infrastructure images had moved to GitHub Container Registry and Harbor would be discontinued.
- For local infra runs, a practical pattern is:
  1. pull `server-lite` / `node-lite` / `ui` from GHCR
  2. retag and push them into a local disposable registry
  3. point `DOCKER_REGISTRY` at that local mirror for `infra.sh`

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
- Harbor is deprecated for infra-image sourcing. Prefer GHCR-backed `server-lite` / `node-lite` / `ui` images or a local mirror of them.
- Keep the Python package version (`VERSION_VANTAGE6`, e.g. `4.14.0`) separate from the Docker image source tags you mirror (e.g. `4.14.0-rc8`).
- Treat amd64 CI as the authoritative signoff environment for infra-backed validation.
- On ARM developer machines, assume the replacement GHCR infra images may require `--platform linux/amd64` when mirroring them locally.
- If `infra.sh up` reports an architecture probe failure or `exec format error`, stop debugging the algorithm itself: the published infra images are not runnable on that host without working amd64 emulation or an amd64 runner.
- For standalone-runtime migrations, prefer deterministic install order:
  1. stable primitives first
  2. git/tar algorithm dependencies with `--no-deps`
  3. target repo install with `--no-deps`
- Do not assume the local workspace repo matches the published pin consumed in tests. Validate the exact installed artifact in a fresh temp venv.
- Avoid reusing a repo-local `.venv` after interrupted pip/self-upgrade failures; create a fresh temp venv instead.
- Do not give `infra.sh` a shared project virtualenv. It upgrades and installs into the environment it is given, so use a disposable `/tmp/...` venv for infra runs.
- If the harness is not checked out as a sibling repo, set `INFRA_DIR` explicitly instead of relying on relative-path defaults.
- If a runtime smoke reports duplicate `run_context` entrypoints after an editable install, check for local `.egg-info` plus installed metadata overlap before changing code.
- For Docker/container smoke, ensure the `RUN_CONTEXT_FILE` input URIs are valid inside the container namespace, not only on the host.

## Recent infra validation notes

- Use the Vantage6 major/minor version that the algorithm release actually targets. Keep the Python package version, e.g. `VERSION_VANTAGE6=4.14.0`, separate from mirrored infrastructure image tags, e.g. `server-lite` / `node-lite` image tag `4.14.0-rc8`.
- GitHub archive downloads can be the bottleneck or failure source during Docker builds. If `pip install ... github.com/.../archive/<sha>.tar.gz` fails with rate limits or codeload errors, build wheels for local checkouts and pass them into Docker with build args instead of changing the pinned release defaults.
- Docker cannot install arbitrary host paths unless the files are inside the build context. For local dependency overrides, copy or build wheels into a repo-local temporary directory that is ignored by git, then pass `/app/<temp-dir>/<wheel>.whl` as the Docker build arg.
- If the algorithm image installs multiple packages that expose the same `run_context` entrypoint group, the master container can fail before doing any work. Inspect the master node's run container traceback first; a duplicate-entrypoint error is an algorithm packaging/runtime issue, not an infra startup issue.
- Use an explicit smoke-test user and log the authenticated organization before creating tasks. If permissions fail, compare against stale server state before assuming different test users have different rights.
- Monitor server and node logs while the smoke is running. Repeated `uWSGI listen queue full` messages can appear during fanout-heavy tests; treat them as load noise only if nodes continue to start containers and task statuses keep advancing. If the server disappears or no task progress follows, treat it as an infra/runtime wedge, especially on ARM hosts running amd64 images through qemu.
- Survival-analysis smoke tests can be much slower than local checks because each phase may fan out through real Vantage6 tasks: validation, imputation metrics, event-time collection, iterations or event-table aggregation, result encryption, and master aggregation.
- If an algorithm phase is known to be slow, record it as performance risk but do not debug it inside an infrastructure lane unless it causes a timeout or incorrect result. Set task timeouts high enough for the current known behavior.
- Always verify that the PR or release branch actually points at the commit that was tested. During GitHub degradation, branch refs and PR synthetic refs may lag or disagree temporarily; use PR metadata or an explicit branch SHA check before tagging.

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
2. Local import gate failed before infra: fix package-root imports and mixed runtime/library boundaries first.
3. Local container `run_context` smoke failed: fix the algorithm/container contract before infra.
4. `up` fails: server/node startup config mismatch.
5. Node logs say `run_context_file currently supports only file-based databases`: generated node config is using a non-file database source while `run_context_file=true`; fix `nodes.env` to `db_type=csv` with a local CSV path, or disable `RUN_CONTEXT_FILE` only for non-run_context algorithms.
6. `run_algo_smoke.sh` fails: algorithm/package/runtime issue. Inspect the master org node container traceback first.
7. `infra.sh test` fails: verify `UI_ENABLED`/`NODES_CONFIG` match how infra was started.
8. Task status `non-existing Docker image`: use local registry with configurable port and retag image.
9. Harbor server/node image pull fails: Harbor is retired for this use; switch to GHCR `server-lite` / `node-lite` / `ui` images or a local mirror of them.
10. Docker build fails while fetching GitHub tarballs: retry only after checking GitHub status; prefer local wheel build-arg overrides for validation during an outage.
11. Master task fails immediately with duplicate `run_context` entrypoints: inspect installed distributions in the image and make the algorithm runtime select its own distribution's entrypoints.
12. Server logs show `uWSGI listen queue full`: check node progress and run-container logs before treating it as fatal; survival fanout can saturate the local demo server while still completing.
13. `down` fails: teardown residue; rerun and inspect remaining containers.
