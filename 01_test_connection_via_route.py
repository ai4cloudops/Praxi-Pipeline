"""Test showing a basic connection to kfp server."""
import os

# from dotenv import load_dotenv

import kfp_tekton

# load_dotenv(override=True)

kubeflow_endpoint="https://ds-pipeline-pipelines-definition-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org"
bearer_token = "sha256~0BJfe202nyTu6hCk35UEGv4Z_-uSrC56KY_6mAo7xDI" # oc whoami --show-token

if __name__ == "__main__":
    client = kfp_tekton.TektonClient(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
    )
    print(client.list_experiments())
