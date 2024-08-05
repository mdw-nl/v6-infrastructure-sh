# Vantage6 Demo Setup Instructions
This guide provides step-by-step instructions to set up a local Vantage6 environment with one server and three nodes (Saturn, Mars, and Jupiter).

## Prerequisites
Ensure you have the following installed on your machine:

- Docker
- Docker Compose
- Python 3 and vantage6

## Setup Instructions

**Step 1: Set Up the Environment**
Run the `create-env.sh` script to set up the virtual environment and install dependencies.

```bash
./create-env.sh
```

**Step 2: Generate Configuration Files**
Run the `generate-configs.sh` script to generate API keys and update the configuration files for the nodes.
```bash
./generate-configs.sh
```

**Step 3: Start the Server**
Navigate to the server-planets directory and start the server using Docker Compose.

```bash
cd server-planets
docker-compose up -d
```

**Step 4: Configure the Server**
Access the UI:

Open your web browser and go to http://localhost:8080.

Login as root:

Use the default root user credentials to log in.

Create Users:

Create the necessary users such as node admins and collaboration admins.
Create a Collaboration:

Create a new collaboration and add the three organizations: Jupiter, Mars, and Saturn.
Generate API Keys:

For each node (Jupiter, Mars, and Saturn), reset or create API keys. These keys will be used for communication between the server and the nodes.

**Step 5: Configure Nodes**
Update Configuration Files:

For each node (Saturn, Mars, Jupiter), update the configuration YAML files to include the correct API keys and paths.

Example for node-saturn/config/saturn.yaml:

```yaml
api_key: <API_KEY_FOR_SATURN>
server_url: http://host.docker.internal
port: 5050
api_path: /api
databases:
  - label: letters
    uri: /path/to/letters.csv  # Update this path to the correct location on your machine
    type: csv
```
Ensure Task Directory Exists:

Ensure that the /tmp folder contains the necessary temporary directories for tasks' data.

```bash
mkdir -p /tmp/vantage6-node-saturn-v6-demo/tasks
mkdir -p /tmp/vantage6-node-mars-v6-demo/tasks
mkdir -p /tmp/vantage6-node-jupiter-v6-demo/tasks
```
**Step 6: Start the Nodes**
Start each node using the v6 node start command with the --attach option for debugging.

Start Saturn Node:

```bash
cd ../node-saturn
v6 node start --config config/saturn.yaml --attach
```
Start Mars Node:

```bash
cd ../node-mars
v6 node start --config config/mars.yaml --attach
```
Start Jupiter Node:

```bash
cd ../node-jupiter
v6 node start --config config/jupiter.yaml --attach
```
**Step 7: Run Jupyter Notebook**
Activate the virtual environment and start Jupyter Notebook.

```bash
source ../venv/bin/activate
jupyter notebook
```

This setup is intended for demo purposes and should help you get acquainted with the Vantage6 platform. It is not per se intended for development or especially production use.