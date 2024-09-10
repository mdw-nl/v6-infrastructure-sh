#! /bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Ensure yq is installed if needed (you can remove this block if not needed anymore)
if ! command_exists yq; then
    echo "Warning: 'yq' is not installed. Skipping its usage."
fi

# Default values for environment variables (can be overridden in CI or locally)
VERSION_VANTAGE6=${VERSION_VANTAGE6:-"4.4.1"}
VENV_PATH=${VENV_PATH:-"./venv"}
SERVER_CONFIG=${SERVER_CONFIG:-"demoserver.yaml"}
ENTITIES_FILE=${ENTITIES_FILE:-"entities.yaml"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"harbor2.vantage6.ai/infrastructure"}
UI_PORT=${UI_PORT:-"80"}
UI_URL=${UI_URL:-"http://localhost"}
SERVER_URL=${SERVER_URL:-"http://localhost:5070"}
API_PATH=${API_PATH:-"/api"}
PYTHON_INTERPRETER=${PYTHON_INTERPRETER:-"python3.11"}

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
        echo "Warning: 'requirements.txt' not found in the current directory."
    else
        echo "Installing dependencies from 'requirements.txt'."
        pip install -r requirements.txt
    fi
    pip install vantage6==$VERSION_VANTAGE6
}

pull_docker_images() {
    echo "Pulling Docker images..."
    docker pull "$DOCKER_REGISTRY/server:$VERSION_VANTAGE6"
    docker pull "$DOCKER_REGISTRY/node:$VERSION_VANTAGE6"
}

start_server() {
    echo "Starting the server..."
    v6 server start --user -c "$(pwd)/$SERVER_CONFIG" --image "$DOCKER_REGISTRY/server:$VERSION_VANTAGE6"
}

import_entities() {
    echo "Importing entities..."
    docker cp "$(pwd)/${ENTITIES_FILE}" vantage6-demoserver-user-ServerType.V6SERVER:/entities.yaml
    docker exec vantage6-demoserver-user-ServerType.V6SERVER /usr/local/bin/vserver-local import --config /mnt/config.yaml /entities.yaml
}

# Function to create and start nodes
start_node() {
    NODE_NAME=$1
    API_KEY=$2
    PORT=$3
    ALGO_DATA_DIRECTORY=${ALGO_DATA_DIRECTORY:-"./data"}
    VANTAGE6_VERSION=$4
    DOCKER_REGISTRY=$5
    SERVER_URL=$6
    VENV_PATH=$7

    # Call the create_node.sh script to create the node and its configuration
    ./create_node.sh "$NODE_NAME" "$API_KEY" "$ALGO_DATA_DIRECTORY" "$PORT" "$VANTAGE6_VERSION" "$DOCKER_REGISTRY" "$SERVER_URL" "$VENV_PATH"
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
start_node "alpha" "844a7d92-1cc9-4856-bf33-0613252d5b3c" "5000" $VERSION_VANTAGE6 $DOCKER_REGISTRY $SERVER_URL $VENV_PATH
start_node "beta" "57143784-19ef-456b-94c9-ba68c8cb079b" "5000" $VERSION_VANTAGE6 $DOCKER_REGISTRY $SERVER_URL $VENV_PATH
start_node "gamma" "57143784-19ef-456b-94c9-ba68c8cb079c" "5000" $VERSION_VANTAGE6 $DOCKER_REGISTRY $SERVER_URL $VENV_PATH

# Start UI and open browser
start_ui
open_browser "$UI_URL"

echo "Setup complete."
