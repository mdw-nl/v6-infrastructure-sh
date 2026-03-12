#!/bin/bash

log() {
  printf '[infra] %s\n' "$*"
}

warn() {
  printf '[infra][warn] %s\n' "$*" >&2
}

fail() {
  printf '[infra][error] %s\n' "$*" >&2
  exit 1
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

parse_bool() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

expand_tilde() {
  local path="$1"
  printf '%s' "${path/#\~/$HOME}"
}

looks_like_uri() {
  local value="$1"
  [[ "$value" == *"://"* ]] || [[ "$value" == sqlite:* ]]
}

abspath_if_local_path() {
  local value="$1"

  if looks_like_uri "$value"; then
    printf '%s' "$value"
    return
  fi

  if [[ "$value" == /* ]]; then
    printf '%s' "$value"
  else
    printf '%s/%s' "$(pwd)" "${value#./}"
  fi
}

ensure_command() {
  local command_name="$1"
  command -v "$command_name" >/dev/null 2>&1 || fail "Required command '$command_name' not found in PATH"
}

init_config_defaults() {
  PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}"
  VERSION_VANTAGE6="${VERSION_VANTAGE6:-4.13.3}"
  ENVIRONMENT="${ENVIRONMENT:-DEV}"

  VENV_PATH="$(expand_tilde "${VENV_PATH:-./venv}")"
  REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

  DOCKER_REGISTRY="${DOCKER_REGISTRY:-harbor2.vantage6.ai/infrastructure}"
  SERVER_URL="${SERVER_URL:-http://localhost:5070}"
  API_PATH="${API_PATH:-/api}"

  SERVER_CONFIG="${SERVER_CONFIG:-./demoserver.yaml}"
  SERVER_CONFIG="$(abspath_if_local_path "$SERVER_CONFIG")"

  local server_config_basename
  server_config_basename="$(basename "$SERVER_CONFIG")"
  SERVER_NAME="${SERVER_NAME:-${server_config_basename%.*}}"

  NODES_CONFIG="${NODES_CONFIG:-./nodes.env}"
  NODES_CONFIG="$(abspath_if_local_path "$NODES_CONFIG")"

  GENERATED_DIR="${GENERATED_DIR:-./generated}"
  GENERATED_DIR="$(abspath_if_local_path "$GENERATED_DIR")"

  ENTITIES_FILE="${ENTITIES_FILE:-$GENERATED_DIR/entities.generated.yaml}"
  ENTITIES_FILE="$(abspath_if_local_path "$ENTITIES_FILE")"

  DATA_DIR_DEFAULT="${DATA_DIR_DEFAULT:-../old/data}"
  DATA_DIR_DEFAULT="$(abspath_if_local_path "$DATA_DIR_DEFAULT")"

  STRICT_DATA_CHECKS="${STRICT_DATA_CHECKS:-true}"
  CLEAN_LOCAL_STATE="${CLEAN_LOCAL_STATE:-true}"
  UI_ENABLED="${UI_ENABLED:-true}"
  UI_PORT="${UI_PORT:-80}"
  UI_URL="${UI_URL:-http://localhost}"

  COLLABORATION_NAME="${COLLABORATION_NAME:-v6-demo}"
}

setup_venv() {
  ensure_command "$PYTHON_INTERPRETER"

  if [ "${RECREATE_ENV:-false}" = true ] && [ -d "$VENV_PATH" ]; then
    log "Removing existing virtual environment at '$VENV_PATH'"
    rm -rf "$VENV_PATH"
  fi

  if [ ! -d "$VENV_PATH" ]; then
    log "Creating virtual environment at '$VENV_PATH' using '$PYTHON_INTERPRETER'"
    "$PYTHON_INTERPRETER" -m venv "$VENV_PATH"
  fi

  # shellcheck source=/dev/null
  . "$VENV_PATH/bin/activate"

  python -m pip install --upgrade pip setuptools wheel
}

install_dependencies() {
  if [ -f "$REQUIREMENTS_FILE" ] && [ -s "$REQUIREMENTS_FILE" ]; then
    log "Installing Python dependencies from '$REQUIREMENTS_FILE'"
    python -m pip install -r "$REQUIREMENTS_FILE"
  else
    log "No non-empty requirements file found at '$REQUIREMENTS_FILE'; skipping"
  fi

  if [ "$VERSION_VANTAGE6" = "latest" ]; then
    python -m pip install vantage6
    VERSION_VANTAGE6="$(python -m pip show vantage6 | awk '/^Version:/ {print $2}')"
  else
    python -m pip install "vantage6==$VERSION_VANTAGE6"
  fi

  log "Using vantage6 version $VERSION_VANTAGE6"
}

pull_docker_images() {
  ensure_command docker

  log "Pulling server and node images from '$DOCKER_REGISTRY'"
  docker pull "$DOCKER_REGISTRY/server:$VERSION_VANTAGE6"
  docker pull "$DOCKER_REGISTRY/node:$VERSION_VANTAGE6"

  if parse_bool "$UI_ENABLED"; then
    docker pull "$DOCKER_REGISTRY/ui"
  fi
}

NODE_NAMES=()
NODE_API_KEYS=()
NODE_DB_URIS=()
NODE_DB_TYPES=()
NODE_DB_LABELS=()

load_node_specs() {
  local specs_file="$NODES_CONFIG"

  [ -f "$specs_file" ] || fail "Node spec file not found: $specs_file"

  NODE_NAMES=()
  NODE_API_KEYS=()
  NODE_DB_URIS=()
  NODE_DB_TYPES=()
  NODE_DB_LABELS=()

  local line_no=0
  while IFS='|' read -r raw_name raw_api_key raw_db_uri raw_db_type raw_db_label raw_extra; do
    line_no=$((line_no + 1))

    local name api_key db_uri db_type db_label
    name="$(trim "${raw_name:-}")"

    if [ -z "$name" ]; then
      continue
    fi

    if [[ "$name" == \#* ]]; then
      continue
    fi

    api_key="$(trim "${raw_api_key:-}")"
    db_uri="$(trim "${raw_db_uri:-}")"
    db_type="$(trim "${raw_db_type:-}")"
    db_label="$(trim "${raw_db_label:-}")"

    if [ -n "$(trim "${raw_extra:-}")" ]; then
      fail "Invalid node spec format at line $line_no in '$specs_file' (too many columns)"
    fi

    [ -n "$api_key" ] || fail "Missing api_key for node '$name' in '$specs_file' (line $line_no)"

    if [ -z "$db_uri" ]; then
      db_uri="$DATA_DIR_DEFAULT/$name.csv"
    fi

    db_type="${db_type:-csv}"
    db_label="${db_label:-default}"

    if ! looks_like_uri "$db_uri"; then
      db_uri="$(abspath_if_local_path "$db_uri")"
    fi

    NODE_NAMES+=("$name")
    NODE_API_KEYS+=("$api_key")
    NODE_DB_URIS+=("$db_uri")
    NODE_DB_TYPES+=("$db_type")
    NODE_DB_LABELS+=("$db_label")
  done < "$specs_file"

  if [ "${#NODE_NAMES[@]}" -eq 0 ]; then
    fail "No node specs found in '$specs_file'"
  fi
}

validate_node_specs() {
  local i
  local strict_data_checks_enabled=false

  if parse_bool "$STRICT_DATA_CHECKS"; then
    strict_data_checks_enabled=true
  fi

  for i in "${!NODE_NAMES[@]}"; do
    local name db_uri db_type
    name="${NODE_NAMES[$i]}"
    db_uri="${NODE_DB_URIS[$i]}"
    db_type="${NODE_DB_TYPES[$i]}"

    if $strict_data_checks_enabled && [ "$db_type" = "csv" ] && ! looks_like_uri "$db_uri"; then
      [ -f "$db_uri" ] || fail "CSV data for node '$name' not found: $db_uri"
    fi
  done
}

print_node_specs() {
  local i
  log "Loaded ${#NODE_NAMES[@]} node specs"
  for i in "${!NODE_NAMES[@]}"; do
    log "- ${NODE_NAMES[$i]} (${NODE_DB_TYPES[$i]}:${NODE_DB_LABELS[$i]}) -> ${NODE_DB_URIS[$i]}"
  done
}

prepare_runtime_dirs() {
  mkdir -p "$GENERATED_DIR"
  mkdir -p "$GENERATED_DIR/nodes"
}

generate_entities_file() {
  local output_file="$ENTITIES_FILE"
  local output_dir
  output_dir="$(dirname "$output_file")"
  mkdir -p "$output_dir"

  {
    echo "collaborations:"
    echo "- encrypted: false"
    echo "  name: $COLLABORATION_NAME"
    echo "  participants:"

    local i
    for i in "${!NODE_NAMES[@]}"; do
      echo "  - api_key: ${NODE_API_KEYS[$i]}"
      echo "    name: ${NODE_NAMES[$i]}"
    done

    echo "nodes: []"
    echo "organizations:"

    for i in "${!NODE_NAMES[@]}"; do
      local name username
      name="${NODE_NAMES[$i]}"
      username="${name}-user"

      echo "- name: $name"
      echo "  address1: ${name} street 1"
      echo "  address2: ''"
      echo "  country: Unknown"
      echo "  domain: ${name}.local"
      echo "  public_key: ''"
      echo "  users:"
      echo "  - email: ${username}@example.org"
      echo "    username: $username"
      echo "    firstname: ${name}"
      echo "    lastname: User"
      echo "    password: ${name}-password"
      echo "  zipcode: '0000AA'"
    done
  } > "$output_file"

  log "Generated entities file at '$output_file'"
}

build_node_config() {
  local node_name="$1"
  local api_key="$2"
  local db_uri="$3"
  local db_type="$4"
  local db_label="$5"
  local output_file="$6"

  cat > "$output_file" <<EOL
api_key: $api_key
api_path: $API_PATH
databases:
  - label: $db_label
    type: $db_type
    uri: $db_uri
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
port: '5070'
server_url: $SERVER_URL
task_dir: ./$node_name/tasks
node_extra_hosts:
  host.docker.internal: host-gateway
EOL
}

start_server() {
  log "Starting server '$SERVER_NAME' using config '$SERVER_CONFIG'"
  v6 server start --user -c "$SERVER_CONFIG" --image "$DOCKER_REGISTRY/server:$VERSION_VANTAGE6"
}

import_entities() {
  local server_container_id
  local attempts=10
  local delay_seconds=1

  while [ "$attempts" -gt 0 ]; do
    server_container_id="$(docker ps -qf "name=^vantage6-${SERVER_NAME}-user" | head -n 1)"
    if [ -n "$server_container_id" ]; then
      break
    fi
    attempts=$((attempts - 1))
    sleep "$delay_seconds"
  done

  [ -n "$server_container_id" ] || fail "Could not find running server container for '$SERVER_NAME'"

  log "Importing entities from '$ENTITIES_FILE'"
  docker cp "$ENTITIES_FILE" "$server_container_id":/entities.yaml

  attempts=5
  while [ "$attempts" -gt 0 ]; do
    if docker exec "$server_container_id" /usr/local/bin/vserver-local import --config /mnt/config.yaml /entities.yaml; then
      return
    fi
    attempts=$((attempts - 1))
    sleep 2
  done

  fail "Could not import entities into server '$SERVER_NAME' after multiple attempts"
}

start_nodes() {
  local i
  for i in "${!NODE_NAMES[@]}"; do
    local node_name api_key db_uri db_type db_label node_config_file
    node_name="${NODE_NAMES[$i]}"
    api_key="${NODE_API_KEYS[$i]}"
    db_uri="${NODE_DB_URIS[$i]}"
    db_type="${NODE_DB_TYPES[$i]}"
    db_label="${NODE_DB_LABELS[$i]}"
    node_config_file="$GENERATED_DIR/nodes/${node_name}.yaml"

    build_node_config "$node_name" "$api_key" "$db_uri" "$db_type" "$db_label" "$node_config_file"

    log "Starting node '$node_name'"
    v6 node start --user -c "$node_config_file"
  done
}

start_ui() {
  if ! parse_bool "$UI_ENABLED"; then
    log "UI is disabled (UI_ENABLED=$UI_ENABLED)"
    return
  fi

  log "Starting UI container on port $UI_PORT"
  docker rm -f vantage6-ui >/dev/null 2>&1 || true

  docker run --rm -d \
    --name vantage6-ui \
    -p "$UI_PORT:$UI_PORT" \
    -e "SERVER_URL=$SERVER_URL" \
    -e "API_PATH=$API_PATH" \
    "$DOCKER_REGISTRY/ui"
}

open_browser() {
  local url="$1"
  case "$(uname -s)" in
    Darwin) open "$url" ;;
    Linux)
      if grep -i microsoft /proc/sys/kernel/osrelease >/dev/null 2>&1; then
        wslview "$url"
      else
        xdg-open "$url"
      fi
      ;;
    *) warn "Automatic browser launch not supported on this OS. Open '$url' manually." ;;
  esac
}

preflight_checks() {
  ensure_command docker
  ensure_command "$PYTHON_INTERPRETER"

  [ -f "$SERVER_CONFIG" ] || fail "Server config not found: $SERVER_CONFIG"
  [ -f "$NODES_CONFIG" ] || fail "Node spec file not found: $NODES_CONFIG"

  docker info >/dev/null 2>&1 || fail "Docker daemon is not reachable"
}

stop_nodes() {
  local i

  for ((i=${#NODE_NAMES[@]}-1; i>=0; i--)); do
    local node_name
    node_name="${NODE_NAMES[$i]}"
    log "Stopping node '$node_name'"
    v6 node stop --user -n "$node_name" >/dev/null 2>&1 || true
  done
}

stop_server() {
  log "Stopping server '$SERVER_NAME'"
  v6 server stop --user -n "$SERVER_NAME" >/dev/null 2>&1 || true
}

remove_containers() {
  log "Removing UI container (if any)"
  docker rm -f vantage6-ui >/dev/null 2>&1 || true

  local server_containers
  server_containers="$(docker ps -aqf "name=^vantage6-${SERVER_NAME}-user")"
  if [ -n "$server_containers" ]; then
    docker rm -f $server_containers >/dev/null 2>&1 || true
  fi

  local i
  for i in "${!NODE_NAMES[@]}"; do
    local node_container
    node_container="vantage6-${NODE_NAMES[$i]}-user"
    docker rm -f "$node_container" >/dev/null 2>&1 || true
  done
}

cleanup_local_state() {
  if ! parse_bool "$CLEAN_LOCAL_STATE"; then
    log "Skipping local state cleanup (CLEAN_LOCAL_STATE=$CLEAN_LOCAL_STATE)"
    return
  fi

  case "$(uname -s)" in
    Darwin)
      rm -rf "$HOME/Library/Application Support/vantage6/node" \
             "$HOME/Library/Application Support/vantage6/server"
      ;;
    Linux)
      rm -rf "$HOME/.local/share/vantage6/node" \
             "$HOME/.local/share/vantage6/server" \
             "$HOME/.cache/vantage6"
      ;;
    *)
      warn "Skipping OS-level cleanup for unsupported platform"
      ;;
  esac
}

expected_container_count() {
  local count=1
  count=$((count + ${#NODE_NAMES[@]}))
  if parse_bool "$UI_ENABLED"; then
    count=$((count + 1))
  fi

  printf '%s' "$count"
}

check_container_presence() {
  ensure_command docker
  docker info >/dev/null 2>&1 || fail "Docker daemon is not reachable"

  local missing=0
  local running
  running="$(docker ps --format '{{.Names}}')"

  if ! echo "$running" | grep -Eq "^vantage6-${SERVER_NAME}-user"; then
    warn "Missing server container with prefix 'vantage6-${SERVER_NAME}-user'"
    missing=1
  fi

  local i
  for i in "${!NODE_NAMES[@]}"; do
    local expected_node
    expected_node="vantage6-${NODE_NAMES[$i]}-user"
    if ! echo "$running" | grep -Fxq "$expected_node"; then
      warn "Missing node container '$expected_node'"
      missing=1
    fi
  done

  if parse_bool "$UI_ENABLED"; then
    if ! echo "$running" | grep -Fxq "vantage6-ui"; then
      warn "Missing UI container 'vantage6-ui'"
      missing=1
    fi
  fi

  if [ "$missing" -ne 0 ]; then
    return 1
  fi

  return 0
}
