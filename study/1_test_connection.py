"""Test showing a basic connection to kfp server."""
import os
import urllib
import requests

import kfp_tekton

kubeflow_endpoint = "https://praxi-kfp-endpoint-praxi.apps.nerc-ocp-test.rc.fas.harvard.edu"
bearer_token = "" # oc whoami --show-token

if __name__ == "__main__":
    client = kfp_tekton.TektonClient(
        host=kubeflow_endpoint,
        existing_token=bearer_token
        ,
        ssl_ca_cert = '/home/ubuntu/Praxi-Pipeline/ca.crt'
    )
    print(client.list_experiments())

