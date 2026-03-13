#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$SCRIPT_DIR/../infrastructure"

cd "$INFRA_DIR"
source ./config.env
source ./functions.sh

init_config_defaults
load_node_specs

echo "Verifying expected container names for this infrastructure config..."

if check_container_presence; then
  echo "All expected containers are present and running."
else
  echo "Error: One or more expected containers are missing."
  exit 1
fi
