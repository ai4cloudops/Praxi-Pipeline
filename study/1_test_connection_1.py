"""Test showing a basic connection to kfp server."""
import os
import urllib
import requests

# from dotenv import load_dotenv

import kfp_tekton

# load_dotenv(override=True)

kubeflow_endpoint = "https://praxi-kfp-endpoint-praxi.apps.nerc-ocp-test.rc.fas.harvard.edu"
bearer_token = "sha256~w3L5oPuscn3hNc9IZFXNJwr1xb9KnTcx462VDTuDseo"

if __name__ == "__main__":
    client = kfp_tekton.TektonClient(
        host=kubeflow_endpoint,
        existing_token=bearer_token
        ,
        ssl_ca_cert = '/home/ubuntu/Praxi-Pipeline/ca.crt'
    )
    print(client.list_experiments())


# r = requests.post(kubeflow_endpoint, verify='/home/ubuntu/Praxi-Pipeline/ca.crt')
# print(r)


# Work with Tekton trigger directly. 


# echo https://$(oc get route ds-pipeline-pipelines-definition -ojsonpath='{.spec.host}')




# the certificate not being accessed by the python code.
# kfp_tekton disable verifiation.
