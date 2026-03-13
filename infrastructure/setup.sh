#!/bin/bash

if [ "${ENVIRONMENT:-}" = "CI" ]; then
  set -e
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ./config.env
source ./functions.sh

RECREATE_ENV=false
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --recreate-env)
      RECREATE_ENV=true
      ;;
    *)
      fail "Unknown parameter: $1"
      ;;
  esac
  shift
done

init_config_defaults
preflight_checks
load_node_specs
validate_node_specs
print_node_specs

setup_venv
install_dependencies
pull_docker_images

prepare_runtime_dirs
generate_entities_file

start_server
import_entities
start_nodes
start_ui

if [ "$ENVIRONMENT" = "DEV" ] && parse_bool "$UI_ENABLED"; then
  log "Opening UI in browser at '$UI_URL'"
  open_browser "$UI_URL"
fi

log "Infrastructure setup complete"
