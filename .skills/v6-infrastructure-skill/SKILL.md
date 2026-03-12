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

`nodes.env` format:

```text
name|api_key|db_uri|db_type|db_label
```

## Local workflow

1. Clone infra harness and pin SHA.
2. Copy algorithm test configs into infra folder.
3. Run lifecycle commands.

```bash
# from algorithm repo root
INFRA_DIR=./tools/v6-infra
git clone git@github.com:mdw-nl/v6-infrastructure-sh.git "$INFRA_DIR"
cd "$INFRA_DIR"
git checkout <INFRA_SHA>

cp ../../tests/infra/config.env infrastructure/config.env
cp ../../tests/infra/nodes.env infrastructure/nodes.env

cd infrastructure
./infra.sh preflight
ENVIRONMENT=CI ./infra.sh up
../../../tests/infra/run_algo_smoke.sh
./infra.sh test
./infra.sh down
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
docker run -d --restart unless-stopped -p 5000:5000 --name v6-local-registry registry:2
docker tag <local-image>:<tag> localhost:5000/<local-image>:<tag>
docker push localhost:5000/<local-image>:<tag>
```

Then use `localhost:5000/<local-image>:<tag>` as the task `image`.

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
  run: ENVIRONMENT=CI tools/v6-infra/infrastructure/infra.sh up

- name: Algorithm smoke
  run: bash tests/infra/run_algo_smoke.sh

- name: Infra smoke
  run: tools/v6-infra/infrastructure/infra.sh test

- name: Shutdown infra
  if: always()
  run: ENVIRONMENT=CI tools/v6-infra/infrastructure/infra.sh down
```

## Guardrails

- Do not duplicate infra scripts into algorithm repos.
- Always pin infra by commit SHA in CI.
- Keep algorithm-specific data and expectations in the algorithm repo only.
- Keep smoke tests deterministic and non-interactive.
- If Docker daemon is unavailable, fail fast at `preflight`.

## Failure triage

1. `preflight` fails: Docker/runtime or missing config paths.
2. `up` fails: server/node startup config mismatch.
3. `run_algo_smoke.sh` fails: algorithm/package/runtime issue.
4. `infra.sh test` fails: container lifecycle issue.
5. `down` fails: teardown residue; rerun and inspect remaining containers.
