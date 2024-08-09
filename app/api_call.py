import requests
import json
import os

def trigger_github_workflow(timestamp):
    # GitHub repository information
    repo_owner = os.getenv('REPO_OWNER')
    repo_name = os.getenv('REPO_NAME')
    workflow_id = 'data-drift-check.yml'  # Adjust if needed
    token = os.getenv('PAT_TOKEN')

    # API endpoint
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches'

    # Payload
    data = {
        'ref': 'main',  # Branch name
        'inputs': {
            'data_timestamp': timestamp.isoformat()  # Use the correct format
        }
    }

    # Headers
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github+json',
        'Content-Type': 'application/json'
    }

    # Make the request
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check response
    if response.status_code == 204:
        print('Workflow triggered successfully.')
    else:
        print(f'Failed to trigger workflow: {response.status_code} - {response.text}')
