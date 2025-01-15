#!/bin/bash

# Usage: create_node.sh <node_name> <api_key> <algo_data_directory> <port> <vantage6_version> <docker_registry> <server_url> <venv_path>

# Arguments
NODE_NAME=$1
API_KEY=$2
ALGO_DATA_DIRECTORY=$3
PORT=$4
VANTAGE6_VERSION=$5
DOCKER_REGISTRY=$6
SERVER_URL=${SERVER_URL:-"http://host.docker.internal"}
VENV_PATH=${VENV_PATH:-"./venv"}


# Check if all arguments are provided
if [ -z "$NODE_NAME" ] || [ -z "$API_KEY" ] || [ -z "$ALGO_DATA_DIRECTORY" ] || [ -z "$PORT" ] || [ -z "$VANTAGE6_VERSION" ] || [ -z "$DOCKER_REGISTRY" ] || [ -z "$SERVER_URL" ] || [ -z "$VENV_PATH" ]; then
    echo "Usage: $0 <node_name> <api_key> <algo_data_directory> <port> <vantage6_version> <docker_registry> <server_url> <venv_path>"
    exit 1
fi

# Create node configuration file
cat <<EOL > ${NODE_NAME}.yaml
api_key: $API_KEY
api_path: /api
databases:
  - label: default
    type: csv
    uri: ${ALGO_DATA_DIRECTORY}/${NODE_NAME}.csv
encryption:
  enabled: false
  private_key: ''
logging:
  backup_count: 5
  datefmt: '%Y-%m-%d %H:%M:%S'
  format: '%(asctime)s - %(name)-14s - %(levelname)-8s - %(message)s'
  level: DEBUG
  loggers:
    - level: warning
      name: urllib3
    - level: warning
      name: requests
    - level: warning
      name: engineio.client
    - level: warning
      name: docker.utils.config
    - level: warning
      name: docker.auth
  max_size: 1024
  use_console: true
port: '$PORT'
server_url: $SERVER_URL
task_dir: ./${NODE_NAME}/tasks
EOL

# Start the node
echo "Starting node '$NODE_NAME'..."

# Activate the virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment at '$VENV_PATH' does not exist."
    exit 1
fi
. "$VENV_PATH/bin/activate"


v6 node start --user -c ${NODE_NAME}.yaml #--image "$DOCKER_REGISTRY/node:$VERSION_VANTAGE6"
