#!/bin/bash
################################################################################
# vantage6 SETUP SCRIPT
#
# This script configures and launches a local vantage6 demo environment, which
# includes:
#   - A vantage6 server
#   - Multiple vantage6 nodes (alpha, beta, gamma)
#   - The vantage6 UI container
#
# By default, it loads environment variables from `config.env`. You can override
# any of these variables (e.g., ENVIRONMENT, VENV_PATH, PYTHON_INTERPRETER)
# on the command line:
#
#     ENVIRONMENT=CI ./setup.sh
#
# Behavior based on ENVIRONMENT:
#   - DEV: does not exit on the first error, and opens the UI in your browser
#   - CI:  exits on the first error (set -e) and does NOT open the browser
#
# Steps performed:
#   1) Create/Activate Python virtual environment
#   2) Install vantage6 dependencies
#   3) Pull Docker images (server, nodes, UI)
#   4) Start vantage6 server, import entities (users/orgs)
#   5) Start vantage6 nodes
#   6) Start vantage6 UI container (and optionally open your web browser if DEV)
#
# For teardown, see `shutdown.sh`.
################################################################################

########################################
# Choose behavior based on ENVIRONMENT #
########################################

# If ENVIRONMENT=CI, enable "exit on error" (set -e).
# If it's DEV (or anything else), don't enforce failing the entire script on error.
if [ "$ENVIRONMENT" = "CI" ]; then
  set -e  # Crash on errors in CI
fi

######################################
# Load configuration and definitions #
######################################

source config.env
source functions.sh

# Echo vantage6 version
echo $VERSION_VANTAGE6

# OS detection
OS=$(uname -s)

# Default value for recreate_env is false
RECREATE_ENV=false

#################
# Parse arguments
#################
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --recreate-env) RECREATE_ENV=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Expand the tilde (~) in VENV_PATH
VENV_PATH="${VENV_PATH/#\~/$HOME}"

##########################
# Main script execution  #
##########################
echo "Setting up the environment..."

setup_venv
install_dependencies
pull_docker_images

start_server
import_entities

# Get absolute data directory path
ALGO_DATA_DIRECTORY="$(get_absolute_filepath "$ALGO_DATA_DIRECTORY")"

# Start nodes
start_node "alpha" "844a7d92-1cc9-4856-bf33-0613252d5b3c" 5070 $ALGO_DATA_DIRECTORY $VERSION_VANTAGE6 $DOCKER_REGISTRY $SERVER_URL $VENV_PATH
start_node "beta" "57143784-19ef-456b-94c9-ba68c8cb079b" 5070 $ALGO_DATA_DIRECTORY $VERSION_VANTAGE6 $DOCKER_REGISTRY $SERVER_URL $VENV_PATH
start_node "gamma" "57143784-19ef-456b-94c9-ba68c8cb079c" 5070 $ALGO_DATA_DIRECTORY $VERSION_VANTAGE6 $DOCKER_REGISTRY $SERVER_URL $VENV_PATH

# Always start the UI container, but only open the browser in DEV
start_ui

if [ "$ENVIRONMENT" = "DEV" ]; then
  echo "Opening the UI in browser..."
  open_browser "$UI_URL"
else
  echo "Skipping browser launch because ENVIRONMENT=\"$ENVIRONMENT\"."
fi

echo "Setup complete."
