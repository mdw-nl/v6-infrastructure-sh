#!/bin/bash

echo "Verifying vantage6 containers..."

# Count how many vantage6 containers are currently running
RUNNING_COUNT=$(docker ps --format '{{.Names}}' | grep '^vantage6-' | wc -l)

# We expect exactly 5 vantage6 containers:
#   - vantage6-demoserver-user-ServerType.V6SERVER
#   - vantage6-alpha-user
#   - vantage6-beta-user
#   - vantage6-gamma-user
#   - vantage6-ui
EXPECTED_COUNT=5

if [ "$RUNNING_COUNT" -eq "$EXPECTED_COUNT" ]; then
  echo "All $EXPECTED_COUNT vantage6 containers are running!"
else
  echo "Error: Expected $EXPECTED_COUNT vantage6 containers, but found $RUNNING_COUNT."
  exit 1
fi
