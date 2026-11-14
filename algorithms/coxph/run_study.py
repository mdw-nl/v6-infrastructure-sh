import json
import sys
from vantage6.client import Client

SERVER_URL = "http://localhost"
SERVER_PORT = 5070
API_PATH = "/api"

USERNAME = "alpha-user"
PASSWORD = "alpha-password"
ALGORITHM_IMAGE = "coxph:latest"
COLLABORATION_NAME = "v6-demo"
INITIATING_ORG = "alpha"

TIME_COL = "Survival.time"
OUTCOME_COL = "deadstatus.event"
# OUTCOME_COL is already 0/1 by default, so OUTCOME_REPRESENTATION can stay
# None. If your data represents the event differently, set
# OUTCOME_REPRESENTATION instead of pre-processing the column, e.g.:
#   OUTCOME_COL = "death_date"
#   OUTCOME_REPRESENTATION = {"mode": "presence"}            # 1 if a death date is recorded, else 0
#   OUTCOME_REPRESENTATION = {"mode": "equals", "positive_value": "Deceased"}
#   OUTCOME_REPRESENTATION = {"mode": "isin", "positive_values": ["Deceased", "Dead"]}
# Add "invert": true to any of the above to flip which side counts as the event.
OUTCOME_REPRESENTATION = None
EXPL_VARS = ["age", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage"]
# Restrict which organizations/nodes central() fans sub-tasks out to. Leave as
# None to target every organization in the collaboration (the default). Set
# this if the collaboration also lists an organization with no node actually
# running -- fanning a sub-task out to it hangs wait_for_results() forever.
#   ORGANIZATIONS = ["alpha", "beta"]
ORGANIZATIONS = None
MAX_ITERATIONS = 10
TOLERANCE = 1e-6

# Minimum participation guards (both no-ops at their defaults, same tier as
# kaplan_meier's min_rows/min_organizations -- requester-side, catches
# misconfiguration, not a node-enforced policy).
MINIMUM_EVENTS = 10
MIN_ORGANIZATIONS = 1

# Risk-set jump guard: if fewer than RISK_SET_MIN_CHANGE individuals left the
# risk set since the last released snapshot, the previous snapshot is held
# instead of releasing a new (thin, individually-recoverable) one.
# MAX_HELD_EVENT_FRACTION aborts the run if too much of the fit ends up
# scored against held snapshots instead of silently degrading it. Both are
# no-ops at their defaults (1, None).
RISK_SET_MIN_CHANGE = 1
MAX_HELD_EVENT_FRACTION = None

# Overfitting guard: errors out instead of fitting if EXPL_VARS is more than
# this percentage of total events across all qualifying nodes (e.g. 10 means
# at least 10 events per covariate required -- the standard Cox
# events-per-variable heuristic). None (the default) disables the check --
# same requester-side tier as the guards above.
MAX_PCT_VARS_VS_OBS = None


def print_model_table(model_json: str) -> None:
    rows = json.loads(model_json)
    coef = rows["Coef"]
    exp_coef = rows["Exp(coef)"]
    se = rows["SE"]
    z = rows["Z"]
    pval = rows["p-value"]
    lower_ci = rows["lower_CI"]
    upper_ci = rows["upper_CI"]

    print(f"\n{'Variable':<25}{'Coef':>10}{'Exp(coef)':>12}{'SE':>10}{'Z':>10}{'p-value':>12}{'95% CI':>22}")
    print("-" * 101)
    for var in coef:
        ci = f"[{lower_ci[var]:.4f}, {upper_ci[var]:.4f}]"
        print(
            f"{var:<25}{coef[var]:>10.4f}{exp_coef[var]:>12.4f}{se[var]:>10.4f}"
            f"{z[var]:>10.4f}{pval[var]:>12.4g}{ci:>22}"
        )


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
    print(f"Time column   : {TIME_COL}")
    print(f"Outcome column: {OUTCOME_COL}  (representation={OUTCOME_REPRESENTATION})")
    print(f"Covariates    : {EXPL_VARS}")
    print(f"Organizations : {ORGANIZATIONS if ORGANIZATIONS else 'all (default)'}")
    print(f"Min organizations: {MIN_ORGANIZATIONS}  (minimum events per org: {MINIMUM_EVENTS})")
    print(f"Risk-set min change: {RISK_SET_MIN_CHANGE}  (max held event fraction: {MAX_HELD_EVENT_FRACTION})")
    print(f"Max pct vars vs obs: {MAX_PCT_VARS_VS_OBS}")
    print()

    task = client.task.create(
        collaboration=collaboration_id,
        organizations=[initiating_org["id"]],
        name="Federated Cox Proportional Hazards",
        description="Compute federated CoxPH hazard ratios across all nodes",
        image=ALGORITHM_IMAGE,
        input_={
            "method": "central",
            "kwargs": {
                "time_col": TIME_COL,
                "outcome_col": OUTCOME_COL,
                "outcome_representation": OUTCOME_REPRESENTATION,
                "expl_vars": EXPL_VARS,
                "organizations": ORGANIZATIONS,
                "max_iterations": MAX_ITERATIONS,
                "tolerance": TOLERANCE,
                "minimum_events": MINIMUM_EVENTS,
                "min_organizations": MIN_ORGANIZATIONS,
                "risk_set_min_change": RISK_SET_MIN_CHANGE,
                "max_held_event_fraction": MAX_HELD_EVENT_FRACTION,
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

    print(f"\nOverall p-value    : {result['overall_p_value']:.4g}")
    print(f"AIC                : {result['aic']:.4f}")
    print(f"Degrees of freedom : {result['degrees_of_freedom']}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
