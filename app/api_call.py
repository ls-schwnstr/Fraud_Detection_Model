import requests
import json
import os
import time


def trigger_github_workflow(timestamp):
    # GitHub repository information
    repo_owner = os.getenv('REPO_OWNER')
    repo_name = os.getenv('REPO_NAME')
    workflow_id = 'data-drift-check.yml'
    token = os.getenv('PAT_TOKEN')
    db_connection_url = ('mssql+pyodbc://<your_username>:<your_password>@<server_name>.database.windows.net:1433/<db_name>?driver=ODBC+Driver+17+for+SQL+Server')

    if not token:
        print("PAT_TOKEN is not set.")
        return

    # API endpoint for triggering the workflow
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches'

    # Payload
    data = {
        'ref': 'main',  # Branch name
        'inputs': {
            'timestamp': timestamp.isoformat(),  # Use the correct format
            'db_connection_url': db_connection_url
        }
    }

    # Headers
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github+json',
        'Content-Type': 'application/json'
    }

    # Trigger the workflow
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check response
    if response.status_code == 204:
        print('Workflow triggered successfully.')
    else:
        print(f'Failed to trigger workflow: {response.status_code} - {response.text}')
        return

    # Polling to check workflow status
    check_workflow_status(repo_owner, repo_name, workflow_id, token)


def check_workflow_status(repo_owner, repo_name, workflow_id, token):
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs?workflow_id={workflow_id}'

    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github+json',
        'Content-Type': 'application/json'
    }

    # Polling loop
    while True:
        # Wait a few seconds before fetching the details to ensure initialization
        time.sleep(20)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            workflow_runs = response.json().get('workflow_runs', [])
            if not workflow_runs:
                print('No workflow runs found.')
                time.sleep(10)  # Wait before retrying
                continue


            if workflow_runs:
                latest_run = workflow_runs[0]
                status = latest_run['status']
                conclusion = latest_run['conclusion']

                print(f'Workflow status: {status}, conclusion: {conclusion}')

                if status == 'completed':
                    if conclusion == 'success':
                        print('Data drift check completed successfully.')
                        return
                    else:
                        print(f'Data drift check failed with conclusion: {conclusion}')
                        return
        else:
            print(f'Failed to retrieve workflow runs: {response.status_code} - {response.text}')

        # Wait for a few seconds before checking again
        time.sleep(30)

