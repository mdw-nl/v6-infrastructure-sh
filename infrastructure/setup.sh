#! /bin/bash

# Load configuration from YAML file
CONFIG_FILE=shell_config.yaml

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Ensure yq is installed
if ! command_exists yq; then
    echo "Error: 'yq' is not installed. Please install it before running this script."
    exit 1
fi

# Parse YAML file
VERSION_NODE=$(yq eval '.version.node' "$CONFIG_FILE")
VERSION_SERVER=$(yq eval '.version.server' "$CONFIG_FILE")
VENV_PATH=$(yq eval '.paths.venv' "$CONFIG_FILE")
SERVER_CONFIG=$(yq eval '.paths.server_config' "$CONFIG_FILE")
ENTITIES_FILE=$(yq eval '.paths.entities_file' "$CONFIG_FILE")
ALPHA_CONFIG=$(yq eval '.paths.alpha_config' "$CONFIG_FILE")
BETA_CONFIG=$(yq eval '.paths.beta_config' "$CONFIG_FILE")
GAMMA_CONFIG=$(yq eval '.paths.gamma_config' "$CONFIG_FILE")
DOCKER_REGISTRY=$(yq eval '.docker.registry' "$CONFIG_FILE")
UI_PORT=$(yq eval '.docker.ui_port' "$CONFIG_FILE")
UI_URL=$(yq eval '.ui.ui_url' "$CONFIG_FILE")
SERVER_URL=$(yq eval '.ui.server_url' "$CONFIG_FILE")
API_PATH=$(yq eval '.ui.api_path' "$CONFIG_FILE")
PYTHON_INTERPRETER=$(yq eval '.python.interpreter' "$CONFIG_FILE")

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

# Functions
setup_venv() {
    # Check if the specified Python interpreter exists
    if ! [ -x "$(command -v "$PYTHON_INTERPRETER")" ]; then
        echo "Error: Specified Python interpreter '$PYTHON_INTERPRETER' not found or not executable."
        exit 1
    fi

    if [ "$RECREATE_ENV" = true ]; then
        echo "Recreating the virtual environment..."
        rm -rf "$VENV_PATH"
    fi

    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment at '$VENV_PATH' using interpreter '$PYTHON_INTERPRETER'."
        "$PYTHON_INTERPRETER" -m venv "$VENV_PATH"
        # Activate and upgrade pip, setuptools, and wheel
        . "$VENV_PATH/bin/activate"
        pip install --upgrade pip setuptools wheel
    else
        echo "Virtual environment already exists at '$VENV_PATH', activating."
        . "$VENV_PATH/bin/activate"
    fi
    echo $(which python)
}

install_dependencies() {
    # Check if requirements.txt exists
    if [ ! -f requirements.txt ]; then
        echo "Error: 'requirements.txt' not found in the current directory."
        exit 1
    fi
    echo "Installing dependencies from 'requirements.txt'."
    pip install -r requirements.txt
}

pull_docker_images() {
    echo "Pulling Docker images..."
    docker pull "$DOCKER_REGISTRY/server:$VERSION_SERVER"
    docker pull "$DOCKER_REGISTRY/node:$VERSION_NODE"
}

start_server() {
    echo "Starting the server..."

    v6 server start --user -c "$(pwd)/$SERVER_CONFIG" --image "$DOCKER_REGISTRY/server:$VERSION_SERVER"
}

import_entities() {
    echo "Importing entities..."
    docker cp "$(pwd)/${ENTITIES_FILE}" vantage6-demoserver-user-ServerType.V6SERVER:/entities.yaml
    docker exec vantage6-demoserver-user-ServerType.V6SERVER /usr/local/bin/vserver-local import --config /mnt/config.yaml /entities.yaml
    # FIXME: maybe #1357 fixes this?
    #v6 server import --wait true --user -c "$(pwd)/$SERVER_CONFIG" "$(pwd)/$ENTITIES_FILE" --image "$DOCKER_REGISTRY/server:$VERSION_SERVER"
}

start_node() {
    local config_file="$1"
    echo "Starting node with config '$config_file'..."
    v6 node start --user -c "$(pwd)/$config_file" --image "$DOCKER_REGISTRY/node:$VERSION_NODE"
}

start_ui() {
    echo "Starting the UI..."
    docker run --rm -d \
        --name vantage6-ui \
        -p "$UI_PORT":"$UI_PORT" \
        -e "SERVER_URL=$SERVER_URL" \
        -e "API_PATH=$API_PATH" \
        "$DOCKER_REGISTRY/ui"
}

open_browser() {
    local url="$1"
    echo "Opening browser at '$url'..."
    case "$OS" in
        Darwin) open "$url" ;;  # macOS
        Linux) xdg-open "$url" ;;  # Linux
        *) echo "Unsupported OS for opening browser automatically. Please open '$url' manually." ;;
    esac
}

# Main script execution
echo "Setting up the environment..."


setup_venv
install_dependencies
pull_docker_images

start_server
import_entities

# Start nodes
start_node "$ALPHA_CONFIG"
start_node "$BETA_CONFIG"
start_node "$GAMMA_CONFIG"

# Start UI and open browser
start_ui
open_browser "$UI_URL"

echo "Setup complete."