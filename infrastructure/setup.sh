#! /bin/bash
#set -e  # Stop on errors

# Load configuration
source config.env
source functions.sh

# The start of the Main script 
echo $VERSION_VANTAGE6

# OS detection
OS=$(uname -s)

# Default value for recreate_env is false
RECREATE_ENV=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --recreate-env) RECREATE_ENV=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Expand the tilde (~) in VENV_PATH
VENV_PATH="${VENV_PATH/#\~/$HOME}"

# Main script execution
echo "Setting up the environment..."

setup_venv
install_dependencies
pull_docker_images

start_server
import_entities

# Start nodes
start_node "alpha" "844a7d92-1cc9-4856-bf33-0613252d5b3c" 5070 $VERSION_VANTAGE6 $DOCKER_REGISTRY $SERVER_URL $VENV_PATH
start_node "beta" "57143784-19ef-456b-94c9-ba68c8cb079b" 5070 $VERSION_VANTAGE6 $DOCKER_REGISTRY $SERVER_URL $VENV_PATH
start_node "gamma" "57143784-19ef-456b-94c9-ba68c8cb079c" 5070 $VERSION_VANTAGE6 $DOCKER_REGISTRY $SERVER_URL $VENV_PATH

# Start UI and open browser
start_ui
open_browser "$UI_URL"

echo "Setup complete."
