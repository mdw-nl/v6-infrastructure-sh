#!/usr/bin/env bash

# Function to generate a random API key
generate_api_key() {
  uuidgen
}

# Directories
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CONFIG_DIR="${BASE_DIR}/node-saturn/config"
SATURN_CONFIG="${CONFIG_DIR}/saturn.yaml"
MARS_CONFIG="${BASE_DIR}/node-mars/config/mars.yaml"
JUPITER_CONFIG="${BASE_DIR}/node-jupiter/config/jupiter.yaml"

# Generate API keys
SATURN_API_KEY=$(generate_api_key)
MARS_API_KEY=$(generate_api_key)
JUPITER_API_KEY=$(generate_api_key)

# Update Saturn configuration
sed -i.bak "s|api_key:.*|api_key: ${SATURN_API_KEY}|" ${SATURN_CONFIG}
echo "Updated Saturn config with API key: ${SATURN_API_KEY}"

# Update Mars configuration
sed -i.bak "s|api_key:.*|api_key: ${MARS_API_KEY}|" ${MARS_CONFIG}
echo "Updated Mars config with API key: ${MARS_API_KEY}"

# Update Jupiter configuration
sed -i.bak "s|api_key:.*|api_key: ${JUPITER_API_KEY}|" ${JUPITER_CONFIG}
echo "Updated Jupiter config with API key: ${JUPITER_API_KEY}"

echo "API keys have been updated in the configuration files."
