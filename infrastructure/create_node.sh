#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ./config.env
source ./functions.sh
init_config_defaults

NODE_NAME="${1:-}"
API_KEY="${2:-}"
DB_URI="${3:-}"
DB_TYPE="${4:-csv}"
DB_LABEL="${5:-default}"

if [ -z "$NODE_NAME" ] || [ -z "$API_KEY" ]; then
  cat <<USAGE >&2
Usage: ./create_node.sh <node_name> <api_key> [db_uri] [db_type] [db_label]
USAGE
  exit 1
fi

if [ -z "$DB_URI" ]; then
  DB_URI="$DATA_DIR_DEFAULT/$NODE_NAME.csv"
fi

if ! looks_like_uri "$DB_URI"; then
  DB_URI="$(abspath_if_local_path "$DB_URI")"
fi

prepare_runtime_dirs
NODE_CONFIG_FILE="$GENERATED_DIR/nodes/${NODE_NAME}.yaml"
build_node_config "$NODE_NAME" "$API_KEY" "$DB_URI" "$DB_TYPE" "$DB_LABEL" "$NODE_CONFIG_FILE"

if [ ! -d "$VENV_PATH" ]; then
  fail "Virtual environment '$VENV_PATH' does not exist. Run ./setup.sh first."
fi

# shellcheck source=/dev/null
. "$VENV_PATH/bin/activate"
v6 node start --user -c "$NODE_CONFIG_FILE"
deactivate

log "Node '$NODE_NAME' started using config '$NODE_CONFIG_FILE'"
