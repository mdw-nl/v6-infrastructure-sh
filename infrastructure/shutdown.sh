#!/bin/bash

if [ "${ENVIRONMENT:-}" = "CI" ]; then
  set -e
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ./config.env
source ./functions.sh

init_config_defaults

if [ -f "$NODES_CONFIG" ]; then
  load_node_specs
else
  warn "Node spec file '$NODES_CONFIG' not found, continuing with container cleanup only"
  NODE_NAMES=()
fi

if [ -d "$VENV_PATH" ]; then
  # shellcheck source=/dev/null
  . "$VENV_PATH/bin/activate"
  stop_nodes
  stop_server
  deactivate
else
  warn "Virtual environment '$VENV_PATH' not found, skipping v6 stop commands"
fi

remove_containers
cleanup_local_state

log "Infrastructure shutdown complete"
