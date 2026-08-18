import base64
import gzip
import json
import sys
from vantage6.client import Client

SERVER_URL = "http://localhost"
SERVER_PORT = 5070
API_PATH = "/api"

USERNAME = "alpha-user"
PASSWORD = "alpha-password"
ALGORITHM_IMAGE = "argos_cnn:latest"
COLLABORATION_NAME = "v6-demo"
INITIATING_ORG = "alpha"

OUTPUT_WEIGHTS_FILE = "argos_cnn_trained_weights.pt"


def main() -> None:
    client = Client(SERVER_URL, SERVER_PORT, API_PATH)

    try:
        client.authenticate(USERNAME, PASSWORD)
    except Exception as exc:
        print(f"Authentication failed: {exc}", file=sys.stderr)
        sys.exit(1)

    client.setup_encryption(None)

    collaborations = client.collaboration.list()
    collaboration = next(
        (c for c in collaborations["data"] if c["name"] == COLLABORATION_NAME), None
    )
    if collaboration is None:
        print(f"Collaboration '{COLLABORATION_NAME}' not found.", file=sys.stderr)
        sys.exit(1)
    collaboration_id = collaboration["id"]

    organizations = client.organization.list()
    initiating_org = next(
        (o for o in organizations["data"] if o["name"] == INITIATING_ORG), None
    )
    if initiating_org is None:
        print(f"Organization '{INITIATING_ORG}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Collaboration : {collaboration['name']} (id={collaboration_id})")
    print(f"Initiating org: {INITIATING_ORG} (id={initiating_org['id']})")
    print(f"Algorithm image: {ALGORITHM_IMAGE}")
    print("Hyperparameters: using argos_cnn.py's central() defaults (N_ROUNDS/LOCAL_STEPS/BATCH_SIZE/LEARNING_RATE)")
    print()

    task = client.task.create(
        collaboration=collaboration_id,
        organizations=[initiating_org["id"]],
        name="Federated ModResNet CT Segmentation",
        description="Train ModResNet across all nodes via FedAvg",
        image=ALGORITHM_IMAGE,
        input_={
            "method": "central",
            # No hyperparameter kwargs here on purpose: central()'s own
            # defaults (N_ROUNDS/LOCAL_STEPS/BATCH_SIZE/LEARNING_RATE in
            # argos_cnn.py) are the single source of truth. Passing copies
            # here would silently override them if the two ever drifted.
            "kwargs": {},
        },
        # Only "default" (the CSV manifest) is requested here — the "nifti"
        # folder database is deliberately not requested, so vantage6's @data()
        # never tries to load it as tabular data; it's just a filesystem
        # mount that nib.load() reads directly (see argos_cnn.py's docstring).
        databases=[{"label": "default"}],
    )
    task_id = task["id"]
    print(f"Task created (id={task_id}), waiting for results...")

    results = client.wait_for_results(task_id)

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

    if "error" in result:
        print(f"Algorithm error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    n_rounds_completed = result.get("n_rounds_completed", 0)
    round_metrics = result.get("round_metrics", [])
    state_dict_b64 = result.get("state_dict")

    print(f"\nRounds completed: {n_rounds_completed}")
    if round_metrics:
        print("\nPer-round weighted dice (weighted by each node's dataset size):")
        for m in round_metrics:
            print(f"  round {m['round']:>2}: n={m['n']:<5} dice={m['dice']:.4f}")

    if not state_dict_b64:
        print("\nNo trained weights returned.")
        return

    # state_dict is gzip+base64 encoded (see argos_cnn.py's _state_dict_to_str) —
    # decode/decompress back to a raw torch.save() blob and write it to disk.
    # Loading it back requires torch: ModResNet().load_state_dict(torch.load(path))
    weights_bytes = gzip.decompress(base64.b64decode(state_dict_b64))
    with open(OUTPUT_WEIGHTS_FILE, "wb") as f:
        f.write(weights_bytes)
    print(f"\nTrained weights written to '{OUTPUT_WEIGHTS_FILE}' ({len(weights_bytes) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
