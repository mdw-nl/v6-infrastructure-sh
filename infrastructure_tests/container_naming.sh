#!/bin/bash

echo "Verifying vantage6 containers..."

# Put the exact container names you expect in an array
EXPECTED_CONTAINERS=(
  "vantage6-demoserver-user-server"
  "vantage6-alpha-user"
  "vantage6-beta-user"
  "vantage6-gamma-user"
  "vantage6-ui"
)

# Get the names of all currently running containers
RUNNING_CONTAINERS=$(docker ps --format "{{.Names}}")

# For each expected container, ensure it's present
for NAME in "${EXPECTED_CONTAINERS[@]}"; do
  if echo "$RUNNING_CONTAINERS" | grep -q "^$NAME\$"; then
    echo "Found container: $NAME"
  else
    echo "Error: Missing container: $NAME"
    exit 1
  fi
done

echo "All vantage6 containers are present and running!"
