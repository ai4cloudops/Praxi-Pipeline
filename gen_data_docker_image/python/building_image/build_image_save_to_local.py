# import rm_image_dockerhub
import subprocess, os, tarfile, shutil, sys, json, yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
sys.path.insert(1, '/home/cc/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/changeset_gen')
import read_layered_image

# The directory containing the Dockerfiles
directory_path = '/home/cc/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles'
# Your Docker Hub username
USERNAME = 'zongshun96'
PASSWORD = ''  # Consider using a Personal Access Token for better security
NAMESPACE = 'zongshun96'
registry_url = 'https://registry.hub.docker.com/v2'
index_url = 'https://index.docker.io/v2'
dockerhub_front_url = 'https://hub.docker.com/v2'
# # /namespaces/{namespace}/repositories/{repository}/tags/{tag}
# def is_tag_pushed(repo_name, tag, token):
#     """Check if the tag is already pushed to Docker Hub."""
#     headers = {'Authorization': f'JWT {token}'}
#     response = requests.get(f"{dockerhub_front_url}/namespaces/{NAMESPACE}/repositories/{repo_name}/tags/{tag}", 
#                             headers=headers)
#     # response = requests.get(f"{dockerhub_front_url}/repositories/{USERNAME}/{repo_name}/tags/{tag}", 
#     #                         headers=headers)

#     # ############## Directly querying the Docker Hub Backend #####################
#     # token = rm_image_dockerhub.get_backend_auth_token(repo_name)
#     # headers = {'Authorization': f'Bearer {token}'}
#     # response = requests.get(f"{index_url}/{USERNAME}/{repo_name}/manifests/{tag}", 
#     #                         headers=headers)
#     # #############################################################################
    
#     return response.status_code == 200

#     # TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:zongshun96/python3_9_18-bookworm.prometheus-flask-exporter_v0_22_4:pull" | jq -r .token)
#     # curl -s -H "Authorization: Bearer $TOKEN" https://index.docker.io/v2/zongshun96/python3_9_18-bookworm.prometheus-flask-exporter_v0_22_4/manifests/TAG

# def all_tag_pushed(dockerfile_path, token, idx):
#     """check all Docker image from a specified Dockerfile."""

#     dockerfile_name = dockerfile_path.name
#     # Create a tag using Docker Hub username and Dockerfile name, excluding 'Dockerfile.' prefix
#     tag = f"{USERNAME}/{dockerfile_name.replace('Dockerfile.', '')}"
#     if is_tag_pushed(dockerfile_name.replace('Dockerfile.', ''), "latest", token):
#         print(f"{idx} Tag {tag} already exists on Docker Hub, skipping push")
#         return True
#     else:
#         print(f"{idx} Tag {tag} miss on Docker Hub")
#         return False

def build_push_remove_docker_image(dockerfile_path, idx):
    """Builds, pushes, and removes a Docker image from a specified Dockerfile."""

    if idx % 10 == 0:
        # Removes all build cache objects.
        cleanup_build_cache()

    dockerfile_name = dockerfile_path.name
    labels_str = dockerfile_name.split(".", 1)[1]
    labels = [pkg_name.replace("_v", "==").replace("_", ".") for pkg_name in dockerfile_name.split(".")[2].split("_p_")]
    # labels = [dockerfile_name.split(".")[2].replace("_v", "==").replace("_", ".")]
    # Create a tag using Docker Hub username and Dockerfile name, excluding 'Dockerfile.' prefix
    tag = f"{USERNAME}/{dockerfile_name.replace('Dockerfile.', '')}"

    # Define the save directory and tar file name
    cwd = "/home/cc/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/changeset_gen/cwd/"
    from pathlib import Path
    if not Path(cwd).exists():
        Path(cwd).mkdir()
    changesets_dir = cwd+"changesets/"
    # changeset_filename_l = read_layered_image.get_free_filename(labels_str, changesets_dir, ".yaml").split(".")
    # # if changeset_filename_l[0].split("-")[1] == "slim"
    if not Path(changesets_dir).exists():
        Path(changesets_dir).mkdir()
    failed_dockerfiles_dir = cwd+"failed_dockerfiles/"
    if not Path(failed_dockerfiles_dir).exists():
        Path(failed_dockerfiles_dir).mkdir()
    save_dir = cwd  # Update this path to your desired directory
    tar_file = f"{save_dir}/{dockerfile_name.replace('Dockerfile.', '')}.tar"
    extracted_dir = f"{save_dir}/extracted/{dockerfile_name.replace('Dockerfile.', '')}"

    build_cmd = ['docker', 'build', '-t', tag, '-f', str(dockerfile_path), str(dockerfile_path.parent)]
    push_cmd = ['docker', 'save', '-o', tar_file, tag]
    remove_cmd = ['docker', 'rmi', tag]
    
    try:
        # Build the Docker image
        subprocess.run(build_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(f"Successfully built {tag}")
        
        # Push the Docker image
        subprocess.run(push_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(f"Successfully pushed {tag}")
        
        # Remove the local Docker image
        subprocess.run(remove_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(f"Successfully removed python3{tag}")

        os.makedirs(extracted_dir, exist_ok=True)
        with tarfile.open(tar_file) as tar:
            tar.extractall(path=extracted_dir)
        os.remove(tar_file)

        # Load the manifest.json file to get the list of layers
        manifest_path = os.path.join(extracted_dir, "manifest.json")
        with open(manifest_path, "r") as manifest_file:
            manifest = json.load(manifest_file)
            layer_files = manifest[0]["Layers"]

        # Initialize a list to hold all filenames
        # changeset = {}

        # Iterate over each layer file listed in the manifest
        # for layer in layer_files:
        layer = layer_files[-1]
        layer_path = os.path.join(extracted_dir, layer)
        # Open the layer tar file
        with tarfile.open(layer_path, "r") as tar:
            # Extract filenames from the tar file
            # for member in tar.getmembers():
            # changeset[layer] = [member.name for member in tar.getmembers()]
            yaml_in = {'labels': labels, 'changes': [member.name for member in tar.getmembers()]}
            changeset_filename = read_layered_image.get_free_filename(labels_str, changesets_dir, ".yaml")
            Path(changeset_filename).touch()
            with open(changeset_filename, 'w') as outfile:
                print("gen_changeset", os.path.dirname(outfile.name))
                print("gen_changeset", changeset_filename)
                yaml.dump(yaml_in, outfile, default_flow_style=False)
        
        shutil.rmtree(extracted_dir)
                    
        # read_layered_image.run(None, labels, None, None, cwd, extracted_dir, labels_str)

        
        

        return f"Done {tag}"
    except subprocess.CalledProcessError as e:
        from pathlib import Path
        target_dockerfile_pathname = failed_dockerfiles_dir+dockerfile_path.name
        if not Path(target_dockerfile_pathname).exists():
            Path(target_dockerfile_pathname).touch()
        with open(target_dockerfile_pathname, "a") as f:
            f.write(e.stderr.decode("utf-8"))
        with open(target_dockerfile_pathname, "a") as f:
            f.write("=========================================\n")
        with open(target_dockerfile_pathname, "a") as f:
            f.write(e.stdout.decode("utf-8"))
        if Path(tar_file).exists():
            os.remove(tar_file)
        if Path(extracted_dir).exists():
            shutil.rmtree(extracted_dir)
        return f"Failed to build/push/remove {tag}: {e}"

def find_dockerfiles(directory):
    """Finds Dockerfiles within the specified directory."""
    path = Path(directory)
    return list(path.glob('Dockerfile.*'))

# def all_tag_pushed_in_parallel(dockerfiles):
#     """check all Docker images in parallel from a list of Dockerfiles."""
#     token = rm_image_dockerhub.get_auth_token(USERNAME, PASSWORD)
#     ret = True
#     with ThreadPoolExecutor(max_workers=1) as executor:
#         future_to_dockerfile = {executor.submit(all_tag_pushed, df, token, idx): df for idx, df in enumerate(dockerfiles)}
#         for idx, future in enumerate(as_completed(future_to_dockerfile)):
#             dockerfile = future_to_dockerfile[future]
#             try:
#                 result = future.result()
#                 ret = ret and result
#                 print(f"{idx} {result}")
#                 if not ret:
#                     return ret
#             except Exception as exc:
#                 print(f'{idx} {dockerfile} generated an exception: {exc}')
#         return ret

def build_push_remove_images_in_parallel(dockerfiles):
    """Builds, pushes, and removes multiple Docker images in parallel from a list of Dockerfiles."""
    # token = rm_image_dockerhub.get_auth_token(USERNAME, PASSWORD)
    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_dockerfile = {executor.submit(build_push_remove_docker_image, df, idx): df for idx, df in enumerate(dockerfiles)}
        for idx, future in enumerate(as_completed(future_to_dockerfile)):
            dockerfile = future_to_dockerfile[future]
            try:
                result = future.result()
                print(f"{idx} {result}")
            except Exception as exc:
                print(f'{idx} {dockerfile} generated an exception: {exc}')


def cleanup_build_cache():
    """Removes all build cache objects."""
    cleanup_cmd = ['docker', 'system', 'prune', '-af']
    try:
        result = subprocess.run(cleanup_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Successfully cleaned up all build cache objects.")
        # print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to clean up build cache objects: {e.stderr.decode()}")


if __name__ == "__main__":
    dockerfiles = find_dockerfiles(directory_path)
    if dockerfiles:
        # print("all tag pushed", all_tag_pushed_in_parallel(dockerfiles))
        build_push_remove_images_in_parallel(dockerfiles)
    else:
        print(f"No Dockerfiles found in {directory_path}.")

    # cleanup_build_cache()


    