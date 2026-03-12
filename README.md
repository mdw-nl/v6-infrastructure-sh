# Vantage6 Local Infrastructure Harness

This repository provides a reusable, config-driven local vantage6 infrastructure for any algorithm package and data layout.

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

## Node spec examples

`infrastructure/nodes.env` supports mixed backends:

```text
alpha|<api_key>|../data/alpha.csv|csv|default
beta|<api_key>|postgresql://user:pass@db:5432/demo|sql|warehouse
```

If `db_uri` is empty, it defaults to `${DATA_DIR_DEFAULT}/<name>.csv`.

## Notes

- Docker daemon must be available before running setup/test.
- `STRICT_DATA_CHECKS=true` enforces local CSV existence checks.
- UI can be disabled with `UI_ENABLED=false`.
