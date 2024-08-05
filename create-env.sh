#!/usr/bin/env bash

VANTAGE6_VERSION=4.5.5

set -euo pipefail

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Creating virtual environment in ${BASE_DIR}/venv"
python3 -m venv "${BASE_DIR}/venv"
source "${BASE_DIR}/venv/bin/activate"

echo "Installing vantage6 version ${VANTAGE6_VERSION}"
pip install vantage6=="${VANTAGE6_VERSION}"
# Install missing dependency
pip install setuptools

echo "Installing vantage6-client version ${VANTAGE6_VERSION}"
pip install vantage6-client==${VANTAGE6_VERSION}

# At the moment, vantage6 config does not support relative paths
# Therefore, we create this quick symlink for demo purposes only
ln -s ${BASE_DIR} /tmp/v6-demo-network
