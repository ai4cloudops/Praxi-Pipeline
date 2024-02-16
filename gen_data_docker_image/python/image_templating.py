from string import Template
from itertools import combinations
from pathlib import Path

def load_package_rank(filter_l=set()):
    import json
    with open('/home/cc/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/top-pypi-packages-30-days.min.json') as f:
        d = json.load(f)
        project_l = []
        for idx, row in enumerate(d["rows"]):
            if row["project"] not in filter_l:
                project_l.append(row["project"])
            if len(project_l) == 1000:
                break
        return project_l

def gen_dockerfile(filter_l=set()):
    all_dep = load_package_rank(filter_l)
    for base_image in ["python:3.9-slim-bullseye", "python:3.9-slim-bookworm", "python:3.9.18-bookworm"]:
        for p_l_idx, (package_chk_l) in enumerate(combinations(all_dep, 1)):

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
            with open('/home/cc/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/DockerfileTemplate.txt', 'r') as file:
                template = Template(file.read())

            # Substitute placeholders with actual values
            dockerfile_content = template.substitute(values)

            # Save the rendered Dockerfile
            save_path = '/home/cc/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles/Dockerfile.'+base_image.replace(":","")+'.'+"-".join(dependencies)
            with open(save_path, 'w') as file:
                file.write(dockerfile_content)

            # print("Dockerfile generated successfully.")


def find_dockerfiles(directory):
    """Finds Dockerfiles within the specified directory."""
    path = Path(directory)
    return list(path.glob('Dockerfile.*'))

if __name__ == "__main__":
    dockerfiles = find_dockerfiles("/home/cc/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles_failed")
    # print([dockerfile.name for dockerfile in dockerfiles])
    filter_l = set([dockerfile.name.split(".")[-1] for dockerfile in dockerfiles])
    # print(filter_l)
    gen_dockerfile(filter_l)