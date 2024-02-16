import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# The directory containing the Dockerfiles
directory_path = '/home/cc/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles'
# Your Docker Hub username
docker_hub_username = 'zongshun96'

def build_push_remove_docker_image(dockerfile_path):
    """Builds, pushes, and removes a Docker image from a specified Dockerfile."""
    dockerfile_name = dockerfile_path.name
    # Create a tag using Docker Hub username and Dockerfile name, excluding 'Dockerfile.' prefix
    tag = f"{docker_hub_username}/{dockerfile_name.replace('Dockerfile.', '')}"
    build_cmd = ['docker', 'build', '-t', tag, '-f', str(dockerfile_path), str(dockerfile_path.parent)]
    push_cmd = ['docker', 'push', tag]
    remove_cmd = ['docker', 'rmi', tag]
    
    try:
        # Build the Docker image
        subprocess.run(build_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully built {tag}")
        
        # Push the Docker image
        subprocess.run(push_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully pushed {tag}")
        
        # Remove the local Docker image
        subprocess.run(remove_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"Successfully removed {tag}"
    except subprocess.CalledProcessError as e:
        from pathlib import Path
        Path('/home/cc/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles_failed/'+dockerfile_path.name).touch()
        return f"Failed to build/push/remove {tag}: {e}"

def find_dockerfiles(directory):
    """Finds Dockerfiles within the specified directory."""
    path = Path(directory)
    return list(path.glob('Dockerfile.*'))

def build_push_remove_images_in_parallel(dockerfiles):
    """Builds, pushes, and removes multiple Docker images in parallel from a list of Dockerfiles."""
    with ThreadPoolExecutor() as executor:
        future_to_dockerfile = {executor.submit(build_push_remove_docker_image, df): df for df in dockerfiles}
        for future in as_completed(future_to_dockerfile):
            dockerfile = future_to_dockerfile[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f'{dockerfile} generated an exception: {exc}')

if __name__ == "__main__":
    dockerfiles = find_dockerfiles(directory_path)
    if dockerfiles:
        build_push_remove_images_in_parallel(dockerfiles)
    else:
        print(f"No Dockerfiles found in {directory_path}.")
