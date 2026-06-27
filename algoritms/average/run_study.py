import json
import sys
from vantage6.client import Client

SERVER_URL = "http://localhost"
SERVER_PORT = 5070
API_PATH = "/api"

USERNAME = "alpha-user"
PASSWORD = "alpha-password"
ALGORITHM_IMAGE = "average:latest"


def main() -> None:
    client = Client(SERVER_URL, SERVER_PORT, API_PATH)

    try:
        client.authenticate(USERNAME, PASSWORD)
    except Exception as exc:
        print(f"Authentication failed: {exc}", file=sys.stderr)
        sys.exit(1)

    client.setup_encryption(None)

    # Resolve collaboration
    collaborations = client.collaboration.list()
    if not collaborations["data"]:
        print("No collaborations found.", file=sys.stderr)
        sys.exit(1)
    collaboration_id = collaborations["data"][0]["id"]

    # Alpha submits the central task — only one organization needed here
    # because the central function itself fans out to all nodes.
    organizations = client.organization.list()
    alpha_org = next(
        (o for o in organizations["data"] if o["name"] == "alpha"), None
    )
    if alpha_org is None:
        print("Organization 'alpha' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Collaboration : {collaborations['data'][0]['name']} (id={collaboration_id})")
    print(f"Initiating org: alpha (id={alpha_org['id']})")
    print(f"Algorithm image: {ALGORITHM_IMAGE}")
    print()

    task = client.task.create(
        collaboration=collaboration_id,
        organizations=[alpha_org["id"]],
        name="Federated Average",
        description="Compute average across all nodes in the collaboration",
        image=ALGORITHM_IMAGE,
        input_={"method": "central"},
        databases=[{"label": "default"}],
    )
    task_id = task["id"]
    print(f"Task created (id={task_id}), waiting for results...")

    results = client.wait_for_results(task_id)

    # wait_for_results returns {'data': [...], 'links': {...}}; extract the items list
    data_items = results.get("data", []) if isinstance(results, dict) else results
    if not data_items:
        print("No results returned.", file=sys.stderr)
        sys.exit(1)

    raw = data_items[0]
    result_str = raw.get("result") if isinstance(raw, dict) else raw
    if not result_str:
        print("Task completed but result was empty (algorithm may have failed).", file=sys.stderr)
        sys.exit(1)

    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    avg = result.get("average")
    variable = result.get("variable", "unknown")
    n = result.get("n", 0)

    if avg is None:
        print(f"Result: no data found for '{variable}' across nodes.")
    else:
        print(f"\nFederated average {variable}: {avg:.4f} (across {n} patients)")


if __name__ == "__main__":
    main()
