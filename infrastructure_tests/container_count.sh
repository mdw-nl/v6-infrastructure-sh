#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$SCRIPT_DIR/../infrastructure"

cd "$INFRA_DIR"
source ./config.env
source ./functions.sh

init_config_defaults
load_node_specs

echo "Verifying expected container count for this infrastructure config..."

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not reachable."
  exit 1
fi

EXPECTED_COUNT="$(expected_container_count)"
ACTUAL_COUNT=0

RUNNING="$(docker ps --format '{{.Names}}')"

if echo "$RUNNING" | grep -Eq "^vantage6-${SERVER_NAME}-user"; then
  ACTUAL_COUNT=$((ACTUAL_COUNT + 1))
fi

for node_name in "${NODE_NAMES[@]}"; do
  if echo "$RUNNING" | grep -Fxq "vantage6-${node_name}-user"; then
    ACTUAL_COUNT=$((ACTUAL_COUNT + 1))
  fi
done

if parse_bool "$UI_ENABLED" && echo "$RUNNING" | grep -Fxq "vantage6-ui"; then
  ACTUAL_COUNT=$((ACTUAL_COUNT + 1))
fi

if [ "$ACTUAL_COUNT" -eq "$EXPECTED_COUNT" ]; then
  echo "Found all expected containers ($ACTUAL_COUNT/$EXPECTED_COUNT)."
else
  echo "Error: Expected $EXPECTED_COUNT containers but found $ACTUAL_COUNT."
  exit 1
fi
