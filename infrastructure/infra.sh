#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

command_name="${1:-help}"
shift || true

case "$command_name" in
  up)
    ./setup.sh "$@"
    ;;
  down)
    ./shutdown.sh "$@"
    ;;
  preflight)
    source ./config.env
    source ./functions.sh
    init_config_defaults
    preflight_checks
    load_node_specs
    validate_node_specs
    print_node_specs
    log "Preflight checks passed"
    ;;
  test)
    "$SCRIPT_DIR/../infrastructure_tests/container_count.sh"
    "$SCRIPT_DIR/../infrastructure_tests/container_naming.sh"
    ;;
  help|--help|-h)
    cat <<USAGE
Usage: ./infra.sh <command> [args]

Commands:
  preflight           Validate local prerequisites and node specs
  up [--recreate-env] Start server, nodes and UI from config
  down                Stop and clean up server/nodes/UI
  test                Run infrastructure smoke tests
USAGE
    ;;
  *)
    echo "Unknown command: $command_name" >&2
    ./infra.sh help
    exit 1
    ;;
esac
