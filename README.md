# Vantage6 Demo Network Infrastructure

This repository provides a demo setup for running a [vantage6](https://vantage6.ai/) server, nodes, and UI locally for testing and development.

## Getting Started

1. **Clone** this repository.
2. **Check** or **edit** `config.env` to set any desired defaults (e.g., vantage6 version, Docker registry, UI port, etc.).
3. **Run** the setup script:
    ```bash
    # Development usage (won't stop on errors, opens browser)
    ENVIRONMENT=DEV ./setup.sh
    ```
   or
    ```bash
    # CI usage (stops on errors, no browser launch)
    ENVIRONMENT=CI ./setup.sh
    ```
    If you omit `ENVIRONMENT=...`, it falls back to whatever is in `config.env`.

4. **Verify** containers are running:
    ```bash
    docker ps
    ```
    You should see the vantage6 server, nodes, and UI container.

5. **Interact** with vantage6 (e.g., run an algorithm). The vantage6 UI can be accessed at http://localhost (configurable in `config.env`).

6. **Stop** and **remove** all containers:
    ```bash
    # Use the same ENVIRONMENT mode you started with, if desired
    ENVIRONMENT=DEV ./shutdown.sh
    ```
    This tears down the vantage6 environment and cleans up leftover files.



