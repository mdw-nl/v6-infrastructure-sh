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
    if [ "$VERSION_VANTAGE6" = "latest" ]; then
        pip install vantage6
        VERSION_VANTAGE6=$(pip show vantage6 | grep Version | awk '{print $2}')
    else
        pip install vantage6==$VERSION_VANTAGE6
    fi
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
    SERVER_CONTAINER_ID=$(docker ps -aqf "name=^vantage6-demoserver")
    docker cp "$(pwd)/${ENTITIES_FILE}" "$SERVER_CONTAINER_ID":/entities.yaml
    docker exec "$SERVER_CONTAINER_ID" /usr/local/bin/vserver-local import --config /mnt/config.yaml /entities.yaml
}

# Function to create and start nodes
start_node() {
    NODE_NAME=$1
    API_KEY=$2
    PORT=$3
    ALGO_DATA_DIRECTORY=$4
    VANTAGE6_VERSION=$5
    DOCKER_REGISTRY=$6
    SERVER_URL=$7
    VENV_PATH=$8

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
        Linux)
            if grep "microsoft" /proc/sys/kernel/osrelease > /dev/null; then # WSL
            wslview "$url"
            else
            xdg-open "$url"
            fi
            ;;
        *) echo "Unsupported OS for opening browser automatically. Please open '$url' manually." ;;
    esac
}

get_absolute_filepath() {
    # 1) Read the input path from the function’s argument
    local input="$1"

    # 2) Compute absolute directory
    local abs_dir
    abs_dir="$(cd "$(dirname "$input")" 2>/dev/null && pwd)"

    # 3) Extract the basename
    local base_name
    base_name="$(basename "$input")"

    # 4) Echo back the fully combined absolute path
    echo "${abs_dir}/${base_name}"
}