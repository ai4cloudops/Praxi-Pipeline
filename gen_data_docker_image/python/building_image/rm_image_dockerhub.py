import image_templating
from itertools import combinations
import requests

# Your Docker Hub credentials or token
USERNAME = 'zongshun96'
PASSWORD = ''  # Consider using a Personal Access Token for better security

def get_backend_auth_token(repo_name):
    """Get an authentication token."""
    url = f'https://auth.docker.io/token?service=registry.docker.io&scope=repository:{USERNAME}/{repo_name}:pull'
    response = requests.get(url)
    response.raise_for_status()  # Raises an error for bad responses
    return response.json()['token']

def get_auth_token(username, password):
    """Get an authentication token."""
    url = 'https://hub.docker.com/v2/users/login/'
    data = {'username': username, 'password': password}
    response = requests.post(url, data=data)
    response.raise_for_status()  # Raises an error for bad responses
    return response.json()['token']

def gen_reponame(all_dep, choose=1, base_images=["python:3.9-slim-bullseye", "python:3.9-slim-bookworm", "python:3.9.18-bookworm"]):
    repo_names = []
    for base_image in base_images:
        for p_l_idx, (package_chk_l) in enumerate(combinations(all_dep, choose)):

            # Assume 'dependency' is now a list of strings
            dependencies = package_chk_l

            repo_names.append(USERNAME+"/"+base_image.replace(":","")+'.'+"-".join(dependencies))
    return repo_names

def delete_repository(token, repository):
    """Delete a specific repository."""
    headers = {'Authorization': f'JWT {token}'}
    repo_url = f'https://hub.docker.com/v2/repositories/{repository}/'
    response = requests.delete(repo_url, headers=headers)
    if response.status_code == 204:
        print(f'Successfully deleted repository: {repository}')
    else:
        print(f'Failed to delete repository: {repository}. Status code: {response.status_code}')

def main(repos):
    token = get_auth_token(USERNAME, PASSWORD)
    for repo in repos:
        delete_repository(token, repo)

if __name__ == '__main__':
    all_dep = image_templating.load_package_rank(count=2000)
    # print(all_dep)
    # all_dep = ["sentence-transformers"]

    repos = gen_reponame(all_dep, base_images=["python:3.12-bookworm"])
    # print(repos)

    main(repos)
