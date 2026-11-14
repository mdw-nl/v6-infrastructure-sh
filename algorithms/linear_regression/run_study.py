import json
import sys
from vantage6.client import Client

SERVER_URL = "http://localhost"
SERVER_PORT = 5070
API_PATH = "/api"

USERNAME = "alpha-user"
PASSWORD = "alpha-password"
ALGORITHM_IMAGE = "linear_regression:latest"
COLLABORATION_NAME = "v6-demo"
INITIATING_ORG = "alpha"

# This is a toy regression to exercise the algorithm against real
# multi-node data (predicting a continuous survival-time column linearly
# isn't a clinically meaningful model) -- point PREDICTOR_COLS/OUTCOME_COL
# at whatever numeric columns your own study needs.
PREDICTOR_COLS = ["age", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage"]
OUTCOME_COL = "Survival.time"
# Restrict which organizations/nodes central() fans sub-tasks out to. Leave as
# None to target every organization in the collaboration (the default). Set
# this if the collaboration also lists an organization with no node actually
# running -- fanning a sub-task out to it hangs wait_for_results() forever.
#   ORGANIZATIONS = ["alpha", "beta"]
ORGANIZATIONS = None

# Minimum participation guards (both no-ops at their defaults, same tier as
# every other algorithm's min_rows/min_organizations -- requester-side,
# catches misconfiguration, not a node-enforced policy).
MIN_ORGANIZATIONS = 1
MIN_ROWS = 0

# Overfitting guard: errors out instead of fitting if the number of
# parameters (predictors plus the intercept) is more than this percentage
# of total usable rows (e.g. 10 means at least 10 rows per parameter
# required). None (the default) disables the check.
MAX_PCT_VARS_VS_OBS = None


def print_model_table(model_json: str) -> None:
    rows = json.loads(model_json)
    coef = rows["Coef"]
    se = rows["SE"]
    tval = rows["t"]
    pval = rows["p-value"]
    lower_ci = rows["lower_CI"]
    upper_ci = rows["upper_CI"]

    print(f"\n{'Variable':<25}{'Coef':>10}{'SE':>10}{'t':>10}{'p-value':>12}{'95% CI':>22}")
    print("-" * 89)
    for var in coef:
        ci = f"[{lower_ci[var]:.4f}, {upper_ci[var]:.4f}]"
        print(f"{var:<25}{coef[var]:>10.4f}{se[var]:>10.4f}{tval[var]:>10.4f}{pval[var]:>12.4g}{ci:>22}")


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
    print(f"Predictors    : {PREDICTOR_COLS}")
    print(f"Outcome       : {OUTCOME_COL}")
    print(f"Organizations : {ORGANIZATIONS if ORGANIZATIONS else 'all (default)'}")
    print(f"Min organizations: {MIN_ORGANIZATIONS}  (min rows per org: {MIN_ROWS})")
    print(f"Max pct vars vs obs: {MAX_PCT_VARS_VS_OBS}")
    print()

    task = client.task.create(
        collaboration=collaboration_id,
        organizations=[initiating_org["id"]],
        name="Federated Linear Regression",
        description="Compute federated multiple linear regression across all nodes",
        image=ALGORITHM_IMAGE,
        input_={
            "method": "central",
            "kwargs": {
                "predictor_cols": PREDICTOR_COLS,
                "outcome_col": OUTCOME_COL,
                "organizations": ORGANIZATIONS,
                "min_organizations": MIN_ORGANIZATIONS,
                "min_rows": MIN_ROWS,
                "max_pct_vars_vs_obs": MAX_PCT_VARS_VS_OBS,
            },
        },
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

    print(f"\nIncluded organizations: {result['included_organizations']}")
    if result["excluded_organizations"]:
        print(f"Excluded organizations: {result['excluded_organizations']}")

    print_model_table(result["model"])

    print(f"\nR-squared          : {result['r_squared']:.4f}")
    print(f"Adjusted R-squared : {result['adjusted_r_squared']:.4f}")
    print(f"Overall p-value    : {result['overall_p_value']:.4g}")
    print(f"AIC                : {result['aic']:.4f}")
    print(f"Degrees of freedom : {result['degrees_of_freedom']}")
    print(f"Observations       : {result['num_observations']}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
