import requests, json
from requests.auth import HTTPBasicAuth
import os

# Configuration for your private Docker registry
registry_url = 'https://registry-route-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org'
username = 'myuser'
password = '!QAZ2wsx'


def get_image_manifest(repository, tag='latest'):
    """ Fetch the image manifest from the private registry using basic auth. """
    headers = {
        'Accept': 'application/vnd.docker.distribution.manifest.v2+json'
    }
    manifest_url = f"{registry_url}/v2/{repository}/manifests/{tag}"
    response = requests.get(manifest_url, headers=headers, auth=HTTPBasicAuth(username, password))
    response.raise_for_status()
    return response.json()

def download_layer(repository, layer_digest, output_dir):
    """ Download an image layer based on its digest from the private registry. """
    layer_url = f"{registry_url}/v2/{repository}/blobs/{layer_digest}"
    try:
        response = requests.get(layer_url, auth=HTTPBasicAuth(username, password), stream=True)
        response.raise_for_status()

        layer_file_path = os.path.join(output_dir, layer_digest.replace(':', '_') + '.tar.gz')
        with open(layer_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return layer_file_path
    except requests.RequestException as e:
        print(f"Failed to download layer {layer_digest}: {str(e)}")
        raise

def download_image(repository, tag, output_dir='/home/cc/tmp'):
    """ Download all layers of a Docker image from a private registry. """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # token = get_auth_token()
    manifest = get_image_manifest(repository, tag)
    with open(f"{output_dir}/manifest.json", 'w') as f:
        json.dump(manifest, f)

    for layer in manifest['layers']:
        digest = layer['digest']
        print(f"Downloading layer {digest}")
        file_path = download_layer(repository, digest, output_dir)
        print(f"Layer downloaded and saved to {file_path}")

# Example usage
if __name__ == '__main__':
    # repository = 'zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0'  # e.g., 'myorg/myimage'
    repository = 'zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4'  # e.g., 'myorg/myimage'
    tag = 'latest'
    try:
        download_image(repository=repository, tag=tag)
        print("Download completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


# import requests
# from requests.auth import HTTPBasicAuth



# # Example usage:
# repository = 'zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0'  # e.g., 'myorg/myimage'
# manifest = get_image_manifest(repository)
# print(manifest)

