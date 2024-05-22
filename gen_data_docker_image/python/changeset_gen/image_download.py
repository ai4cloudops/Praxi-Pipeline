import requests
import json

# Replace these variables with your details
username = 'YOUR_USERNAME'
password = 'YOUR_PASSWORD'
repository = 'REPOSITORY'
tag = 'TAG'

def get_token(repo_name):
    """
    Authenticate with Docker Hub and retrieve an access token.
    """
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo_name}:pull"
    response = requests.get(url, auth=(username, password))
    response.raise_for_status()
    return response.json()["token"]

def get_manifest(repo_name, tag, token):
    """
    Get the manifest for a specific tag of a repository.
    """
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.docker.distribution.manifest.v2+json'
    }
    url = f"https://index.docker.io/v2/{repo_name}/manifests/{tag}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def download_layer(repo_name, layer_digest, token, layer_number):
    """
    Download a specific layer of the image.
    """
    headers = {'Authorization': f'Bearer {token}'}
    url = f"https://index.docker.io/v2/{repo_name}/blobs/{layer_digest}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    filename = f"layer_{layer_number}.tar"
    with open(filename, 'wb') as file:
        file.write(response.content)
    print(f"Layer {layer_number} downloaded as {filename}")

def main():
    token = get_token(repository)
    manifest = get_manifest(repository, tag, token)

    for i, layer in enumerate(manifest['layers']):
        digest = layer['digest']
        download_layer(repository, digest, token, i)

if __name__ == "__main__":
    main()
