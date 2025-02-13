import json
from vantage6.client import Client
from algorithms.config import config

def run_task(algorithm_config):
    # Initialize client and authenticate using our config object
    client = Client(config.server_url, config.server_port, config.server_api)
    client.authenticate(username=config.username, password=config.password)
    if "organization_key" in config:
        client.setup_encryption(config.organization_key)

    # Retrieve organizations and perform an intermediary check
    organizations_data = client.organization.list().get('data', [])
    organizations = {org['name']: org['id'] for org in organizations_data}
    if not organizations:
        raise RuntimeError("No organizations found!")
    
    # Retrieve collaborations and check if they exist
    collaborations = client.collaboration.list().get('data', [])
    if not collaborations:
        raise RuntimeError("No collaborations found!")

    # Create and run the task
    task = client.task.create(
        image=algorithm_config['image'],
        name=algorithm_config['name'],
        description=algorithm_config['description'],
        input_=algorithm_config['input'],
        organizations=algorithm_config['organizations'],
        collaboration=algorithm_config['collaboration'],
        databases=algorithm_config['databases']
    )
    
    task_id = task.get("id")
    client.wait_for_results(task_id)
    results = client.result.get(task_id)
    return results

if __name__ == "__main__":
    # Define algorithm-specific configurations as dictionaries.
    km_config = {
        'image': 'harbor2.vantage6.ai/algorithms/kaplan-meier',
        'name': 'demo-km-analysis',
        'description': 'Kaplan-Meier dry-run',
        'input': {
            'method': 'kaplan_meier_central',
            'kwargs': {
                'time_column_name': 'Survival.time',
                'censor_column_name': 'deadstatus.event',
                'organizations_to_include': [1, 2, 3]
            }
        },
        'organizations': [2],
        'collaboration': 1,
        'databases': [{'label': 'default'}]
    }

    avg_config = {
        'image': 'ghcr.io/mdw-nl/v6-average-py:v1.0.1',
        'name': 'demo-average',
        'description': 'Average dry-run',
        'input': {
            'method': 'central_average',
            'kwargs': {
                'column_name': ['age'],
                'org_ids': [1, 2, 3]
            }
        },
        'organizations': [2],
        'collaboration': 1,
        'databases': [{'label': 'default'}]
    }

    # Run the tasks. You can choose to run one or both algorithms.
    results_km = run_task(km_config)
    results_avg = run_task(avg_config)

    # Instead of printing results directly, we collect them in a JSON dictionary.
    output = {
        "km_results": results_km,
        "average_results": results_avg
    }

    print(json.dumps(output, indent=2))
