import tarfile, sys, io, json, os, tempfile, subprocess, yaml, pickle, shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from itertools import combinations
# import requests
from pathlib import Path
from pprint import pprint
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/cc/Praxi-Pipeline/gen_data_docker_image/python/templating_image/')
import image_templating

def get_free_filename(stub, directory, suffix=''):
    """ Get a file name that is unique in the given directory
    input: the "stub" (string you would like to be the beginning of the file
        name), the name of the directory, and the suffix (denoting the file type)
    output: file name using the stub and suffix that is currently unused
        in the given directory
    """
    counter = 0
    while True:
        file_candidate = '{}{}.{}{}'.format(
            str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            counter += 1
        else:  # No match found
            # print("get_free_filename no suffix")
            # Path(file_candidate).touch()
            return file_candidate, counter==0

def gen_dockerfile(all_dep, choose=1, base_images=["python:3.9.18-bullseye", "python:3.9-slim-bullseye", "python:3.9-slim-bookworm", "python:3.9.18-bookworm"], p_l_len = -1): # python:3.12-bookworm
    images_l, labels_l, base_image_l = [], [], []
    for p_l_idx, (package_chk_l) in enumerate(combinations(all_dep, choose)):
        if p_l_idx == p_l_len:
            break
        for base_image in base_images:

            # Avoid putting same packages in same images
            diff_packages_set = set([package_chk.split("==")[0] for package_chk in package_chk_l])
            if len(diff_packages_set) != choose:
                continue

            # Assume 'dependency' is now a list of strings
            dependencies = package_chk_l

            # Generate data pairs, i.e., images and labels
            dependencies_str = (base_image.replace(":","")).replace(".","_")+'.'+("-".join([dep.replace("==", "_v") for dep in dependencies])).replace(".","_")
            images_l.append(dependencies_str)
            labels_l.append(package_chk_l)
            base_image_l.append(base_image)

            # print("Dockerfile generated successfully.")
    return images_l, labels_l, base_image_l

def pull_save_image(image_name, labels, base_image, src, cwd):
    # image_name = "zongshun96/introspected_container:0.01"
    labels_str = (base_image.replace(":","")).replace(".","_")+'.'+("-".join([dep.replace("==", "_v") for dep in labels])).replace(".","_")


    # 'bash /home/cc/Praxi-Pipeline/get_layer_changes/src/download-frozen-image-v2.sh                       /home/cc/Praxi-Pipeline/get_layer_changes/cwd/introspected_container                            zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0:latest'
    # 'bash /home/cc/Praxi-Pipeline/gen_data_docker_image/python/changeset_gen/download-frozen-image-v2.sh  /home/cc/Praxi-Pipeline/gen_data_docker_image/python/changeset_gen/cwd/introspected_container   python3_9_18-bullseye.ruff_v0_2_2'
    cmd1 = "bash "+src+"download-frozen-image-v2.sh "+cwd+labels_str+" "+"zongshun96/"+image_name+":latest"
    # p_cmd1 = subprocess.Popen(cmd1.split(" "), stdin=subprocess.PIPE)
    # p_cmd1.communicate()
    try:
        p_cmd1 = subprocess.run(cmd1.split(" "), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Failed: {cmd1}: {e}")
        return None

    image_layer_dir = cwd+labels_str
    return image_layer_dir, labels_str

def run(image_name, labels, base_image, src, cwd, image_layer_dir=None, labels_str=None):

    if not image_layer_dir:
        image_layer_dir, labels_str = pull_save_image(image_name, labels, base_image, src, cwd)

    image_d = {}
    image_meta_d = {}
    for root, subdirs, files in os.walk(image_layer_dir):
        for file_name in files:
            print(os.path.join(root, file_name))
            if file_name == "manifest.json":
                with open(os.path.join(root, file_name), "r") as json_file:
                    content = json.load(json_file)
                    image_meta_d[file_name] = content
            elif file_name == "json":
                # json_file = tar.extractfile(member)
                with open(os.path.join(root, file_name), "r") as json_file:
                    content = json.load(json_file)
                    image_meta_d[root] = content
            elif file_name[-4:] == "json":
                # json_file = tar.extractfile(member)
                with open(os.path.join(root, file_name), "r") as json_file:
                    content = json.load(json_file)
                    image_meta_d[file_name] = content
            elif file_name[-3:] == "tar":
                # tar_bytes = io.BytesIO(tar.extractfile(member).read())
                tar_file = os.path.join(root, file_name)
                inner_tar = tarfile.open(tar_file)
                image_d[root.split("/")[-1]] = inner_tar.getnames()
                inner_tar.close()
    shutil.rmtree(image_layer_dir)


    changesets_l = []             
    changesets_dir = cwd+"changesets/"
    if not Path(changesets_dir).exists():
        Path(changesets_dir).mkdir()
        # os.chmod(changesets_dir, 777)
    # pprint(image_d)
    # pprint(image_meta_d)

    for image_manifest in image_meta_d["manifest.json"]:
        image_config_name = image_manifest["Config"]
        image_config_history_iter = iter(image_meta_d[image_config_name]["history"])
        
        for layer in image_manifest["Layers"]:
            try:
                image_config_history = next(image_config_history_iter)
                while 'empty_layer' in image_config_history:
                    print(image_config_history, "skipped")
                    image_config_history = next(image_config_history_iter)
            except StopIteration:
                print("image_config_history is None")
                sys.exit(-1)
            print(image_config_history)

            # yaml_in = {'open_time': open_time, 'close_time': close_time, 'label': label, 'changes': changes}
            yaml_in = {'labels': list(labels), 'changes': image_d[layer.split("/")[0]]}
            changeset_filename, new_sample_bool = get_free_filename(labels_str, changesets_dir, ".yaml")
            Path(changeset_filename).touch()
            with open(changeset_filename, 'w') as outfile:
                print("gen_changeset", os.path.dirname(outfile.name))
                print("gen_changeset", changeset_filename)
                yaml.dump(yaml_in, outfile, default_flow_style=False)
            changesets_l.append(yaml_in)
                
    return changesets_l


def download_images_in_parallel(images_l, labels_l, base_image_l, src, cwd):
    """Download multiple Docker images in parallel from a list of Dockerfiles."""
    # for image_name, labels, base_image in zip(images_l, labels_l, base_image_l):
    #     changesets_l = run(image_name, labels, base_image, src, cwd)
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_dockerfile = {executor.submit(run, image_name, labels, base_image, src, cwd): (base_image.replace(":","")).replace(".","_")+'.'+("-".join([dep.replace("==", "_v") for dep in labels])).replace(".","_")
                                 for image_name, labels, base_image in zip(images_l, labels_l, base_image_l)}
        for idx, future in enumerate(as_completed(future_to_dockerfile)):
            dockerfile = future_to_dockerfile[future]
            try:
                result = future.result()
                # print(f"{idx} {result}")
            except Exception as exc:
                print(f'{idx} {dockerfile} generated an exception: {exc}')

if __name__ == "__main__":
    dockerfiles_failed_dir = "/home/cc/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles_failed"
    if not Path(dockerfiles_failed_dir).exists():
        Path(dockerfiles_failed_dir).mkdir()
    dockerfiles = image_templating.find_dockerfiles(dockerfiles_failed_dir)
    # print([dockerfile.name for dockerfile in dockerfiles])

    filter_l = set([dockerfile.name.split(".")[-1] for dockerfile in dockerfiles])
    # print(filter_l)

    all_dep = image_templating.load_package_rank(filter_l)
    # print(all_dep)

    package_versions = image_templating.fetch_versions_for_multiple_packages(all_dep)
    # # Print the versions
    # for package_name, versions in package_versions.items():
    #     print(f"{package_name}: {versions}")

    all_dep_with_ver = image_templating.format_dep_with_versions(package_versions)
    # print(all_dep_with_ver)

    images_l, labels_l, base_image_l = gen_dockerfile(all_dep_with_ver)
    # print(images_l)





    homed = "/home/cc/Praxi-Pipeline/gen_data_docker_image/python/changeset_gen/"
    # homed = "/pipelines/component/"
    src = homed
    if not Path(src).exists():
        Path(src).mkdir()
        # os.chmod(src, 777)
    cwd = homed+"cwd/"
    if not Path(cwd).exists():
        Path(cwd).mkdir()
        # os.chmod(cwd, 777)


    download_images_in_parallel(images_l, labels_l, base_image_l, src, cwd)

    # for image_name, labels, base_image in zip(images_l, labels_l, base_image_l):
    #     changesets_l = run(image_name, labels, base_image, src, cwd)
        # cs_dump_path = cwd+"changesets_l_dump"
        # # cs_path = "/home/cc/Praxi-Pipeline/get_layer_changes/cwd/unknown"
        # with open(cs_dump_path, 'wb') as writer:
        #     pickle.dump(changesets_l, writer)
        # # for idx, changeset in enumerate(changesets_l):
        # #     with open(cs_path+"-{%d}.yaml".format(idx), 'w') as writer:
        # #         yaml.dump(changeset, writer, default_flow_style=False)
        #     break