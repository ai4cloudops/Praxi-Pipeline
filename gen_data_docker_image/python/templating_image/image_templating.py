import pickle
from string import Template
from itertools import combinations
import collections
from pathlib import Path
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

def list_package_versions(package_name):
    try:
        # Run the pip command and capture the output
        result = subprocess.run(['pip', 'index', 'versions', package_name], check=True, text=True, capture_output=True)
        
        # Use a regular expression to find versions in the output
        # print(result.stdout)
        versions = re.findall(r'(\d+\.\d+(?:\.\d+)*)', result.stdout)[1:-2]
        
        return package_name, versions
    except subprocess.CalledProcessError:
        # If pip command fails, return package name and an empty list
        return package_name, []

def fetch_versions_for_multiple_packages(package_names):
    # Dictionary to hold package versions
    package_versions = {}
    
    # Use ThreadPoolExecutor to parallelize requests
    with ThreadPoolExecutor(max_workers=192) as executor:
        # Submit all the tasks and get a list of Future objects
        future_to_package = {executor.submit(list_package_versions, pkg): pkg for pkg in package_names}
        
        # As each future completes, update the dictionary
        for future in as_completed(future_to_package):
            package_name, versions = future.result()
            package_versions[package_name] = versions
            
    return package_versions


def load_package_rank(filter_l=set(),count=1000):
    import json
    with open('/home/cc/Praxi-Pipeline/gen_data_docker_image/python/templating_image/top-pypi-packages-30-days.min.json') as f:
        d = json.load(f)
        project_l = set()
        for idx, row in enumerate(d["rows"]):
            if row["project"] not in filter_l:
                if row["project"] not in project_l:
                    project_l.add(row["project"])
                else:
                    print(row["project"])
            if len(project_l) == count:
                break
        return list(project_l)

def gen_dockerfile(all_dep, choose=1, base_images=["python:3.9.18-bullseye", "python:3.9-slim-bullseye", "python:3.9-slim-bookworm", "python:3.9.18-bookworm"]): # python:3.12-bookworm
    seen = list()
    for p_l_idx, (package_chk_l) in enumerate(combinations(all_dep, choose)):
        if p_l_idx == -1:
            break
        for base_image in base_images:

            # Avoid putting same packages in same images
            diff_packages_set = set([package_chk.split("==")[0] for package_chk in package_chk_l])
            if len(diff_packages_set) != choose:
                continue

            # Assume 'dependency' is now a list of strings
            dependencies = package_chk_l

            # Convert the list of dependencies into a single string
            dependencies_str = ' '.join(dependencies)

            # Define the values for the placeholders, including the newly formatted dependency string
            values = {
                'base_image': base_image,
                'dependencies': dependencies_str
            }

            # Load the template
            with open('/home/cc/Praxi-Pipeline/gen_data_docker_image/python/templating_image/DockerfileTemplate.txt', 'r') as file:
                template = Template(file.read())

            # Substitute placeholders with actual values
            dockerfile_content = template.substitute(values)

            # Save the rendered Dockerfile
            dependencies_str = (base_image.split("/")[-1].replace(":","")).replace(".","_")+'.'+("_p_".join([dep.replace("==", "_v") for dep in dependencies])).replace(".","_")
            save_path = '/home/cc/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles/Dockerfile.'+dependencies_str
            with open(save_path, 'w') as file:
                file.write(dockerfile_content)
            # if save_path in seen:
            #     print(save_path)
            seen.append(package_chk_l)

            # print("Dockerfile generated successfully.")
    return seen

def format_dep_with_versions(package_versions, filter_l=set()):
    all_dep_with_ver = set()
    for package_name, versions in package_versions.items():
        count = 0
        for version in versions:
            formatted_version = f"{package_name}=={version}"
            if formatted_version not in all_dep_with_ver and formatted_version.replace("==", "_v").replace(".","_") not in filter_l:
                all_dep_with_ver.add(formatted_version)
                count += 1
                if count == 3:
                    break
        if count < 3:
            print("count:", count,package_name, versions)
            Path('/home/cc/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles_failed/'+'Dockerfile..'+package_name).touch()
    # print(all_dep_with_ver)
    return all_dep_with_ver

def find_dockerfiles(directory):
    """Finds Dockerfiles within the specified directory."""
    path = Path(directory)
    return list(path.glob('Dockerfile.*'))

if __name__ == "__main__":
    # dockerfiles_failed_dir = "/home/cc/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles_failed"
    # if not Path(dockerfiles_failed_dir).exists():
    #     Path(dockerfiles_failed_dir).mkdir()
    # dockerfiles = find_dockerfiles(dockerfiles_failed_dir)
    # # print([dockerfile.name for dockerfile in dockerfiles])

    dockerfiles_dir = "/home/cc/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles"
    if not Path(dockerfiles_dir).exists():
        Path(dockerfiles_dir).mkdir()

    # filter_l = set([dockerfile.name.split(".")[-1] for dockerfile in dockerfiles])
    # # print(filter_l)

    # all_dep = load_package_rank(filter_l)
    # # print(all_dep)

    # package_versions = fetch_versions_for_multiple_packages(all_dep)
    # # # Print the versions
    # # for package_name, versions in package_versions.items():
    # #     print(f"{package_name}: {versions}")

    # all_dep_with_ver = format_dep_with_versions(package_versions, filter_l)

    index_label_mapping_path = "/home/cc/Praxi-Pipeline/data/data4/index_label_mapping"
    # with open(index_label_mapping_path, 'wb') as fp:
    #     all_label_l = pickle.load(fp)
    with open(index_label_mapping_path, 'rb') as fp:
        all_dep_with_ver = set(pickle.load(fp))

    # saved = gen_dockerfile(list(all_dep_with_ver), choose=1, base_images=["public.ecr.aws/docker/library/python:3.9.18-bullseye", "public.ecr.aws/docker/library/python:3.9-slim-bullseye", "public.ecr.aws/docker/library/python:3.9-slim-bookworm", "public.ecr.aws/docker/library/python:3.9.18-bookworm"])
    saved = gen_dockerfile(list(all_dep_with_ver), choose=2)

    # index_label_mapping_path = "/home/cc/Praxi-Pipeline/data/data4/index_label_mapping"
    # # with open(index_label_mapping_path, 'wb') as fp:
    # #     all_label_l = pickle.load(fp)
    # with open(index_label_mapping_path, 'rb') as fp:
    #     clf_labels_l = pickle.load(fp)

    with open(f"{dockerfiles_dir}/inventory.json", "w") as outfile:
        json.dump(saved, outfile)
    # print(saved)

    