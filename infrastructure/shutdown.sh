#!/bin/bash
################################################################################
# vantage6 SHUTDOWN SCRIPT
#
# This script stops and cleans up a vantage6 demo environment.
# It:
#   1. Loads environment variables from `config.env` (unless already set),
#   2. Activates the vantage6 virtual environment,
#   3. Stops all vantage6 nodes (alpha, beta, gamma) and server,
#   4. Removes vantage6 Docker containers (server, nodes, UI),
#   5. Cleans vantage6 caches/state files (OS-specific).
#
# Usage:
#   ENVIRONMENT=CI ./shutdown.sh
#   or
#   ./shutdown.sh    (defaults to the ENVIRONMENT specified in config.env)
################################################################################

##
# 1) [Optional] If ENVIRONMENT=CI, stop immediately on any error
##
[ "$ENVIRONMENT" = "CI" ] && set -e

##
# 2) Source default environment variables from `config.env`.
#    - Using `${VAR:-default}` lets you override them via the command line:
#         ENVIRONMENT=CI ./shutdown.sh
#      which takes precedence over config.env
##
source config.env

# Expand the tilde (~) in VENV_PATH
VENV_PATH="${VENV_PATH/#\~/$HOME}"

##
# 3) Activate vantage6 virtual environment (if it exists)
##
if [ ! -d "$VENV_PATH" ]; then
    echo "Warning: Virtual environment at '$VENV_PATH' does not exist."
    echo "Skipping vantage6 server/node shutdown commands."
else
    echo "Activating virtual environment at '$VENV_PATH'..."
    . "$VENV_PATH/bin/activate"

    # Define functions to stop vantage6 services
    stop_node() {
        local node_name="$1"
        echo "Stopping node '$node_name'..."
        v6 node stop --user -n "$node_name"
    }

    stop_server() {
        local server_name="$1"
        echo "Stopping server '$server_name'..."
        v6 server stop --user -n "$server_name"
    }

    # Stop vantage6 nodes and server
    stop_node "gamma"
    stop_node "beta"
    stop_node "alpha"
    stop_server "demoserver"

    # Deactivate the virtual environment
    deactivate
fi

##
# 4) Remove vantage6 Docker containers
##
echo "Removing vantage6 UI container (if running)..."
docker stop vantage6-ui 2>/dev/null || true
docker rm -f vantage6-ui 2>/dev/null || true

echo "Removing vantage6 server/node containers (if running)..."
SERVER_CONTAINER_ID=$(docker ps -aqf "name=^vantage6-demoserver")
docker stop "$SERVER_CONTAINER_ID" 2>/dev/null || true
docker rm -f "$SERVER_CONTAINER_ID" 2>/dev/null || true
docker rm -f vantage6-alpha-user vantage6-beta-user vantage6-gamma-user 2>/dev/null || true

##
# 5) Clean vantage6 files from OS-specific paths
##
case "$(uname -s)" in
    Darwin)  # macOS
        echo "Removing vantage6 data from macOS paths..."
        rm -rf "$HOME/Library/Application Support/vantage6/node" \
               "$HOME/Library/Application Support/vantage6/server"
        ;;
    Linux)   # Linux
        echo "Removing vantage6 data from Linux paths..."
        rm -rf "$HOME/.local/share/vantage6/node" \
               "$HOME/.local/share/vantage6/server" \
               "$HOME/.cache/vantage6"
        ;;
    *)
        echo "Unsupported OS for vantage6 data cleanup."
        ;;
esac

echo "Shutdown complete."
