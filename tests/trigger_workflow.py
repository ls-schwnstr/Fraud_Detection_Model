import requests
import os

def trigger_workflow():
    github_token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Replace with your repo information
    repo = "your-username/your-repo"
    workflow_id = "data_drift_workflow.yml"  # The filename of your workflow
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}/dispatches"

    # Replace 'main' with the branch you want to trigger the workflow on
    payload = {
        "ref": "main",
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 204:
        print("Workflow triggered successfully")
    else:
        print(f"Failed to trigger workflow: {response.status_code} - {response.text}")

if __name__ == "__main__":
    trigger_workflow()
