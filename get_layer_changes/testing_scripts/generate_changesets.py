import shutil
import tarfile, sys, io, json, os, tempfile, subprocess, yaml, pickle
from collections import defaultdict
# import requests
from pathlib import Path
from pprint import pprint
sys.path.insert(0, "/home/cc/Praxi-study/Praxi-Pipeline/get_layer_changes/src")
import image_downloader

def get_free_filename(stub, directory, suffix=''):
    """ Get a file name that is unique in the given directory
    input: the "stub" (string you would like to be the beginning of the file
        name), the name of the directory, and the suffix (denoting the file type)
    output: file name using the stub and suffix that is currently unused
        in the given directory
    """
    counter = 0
    while True:
        file_candidate = '{}{}-{}{}'.format(
            str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            counter += 1
        else:  # No match found
            print("get_free_filename no suffix")
            Path(file_candidate).touch()
            return file_candidate


homed = "/home/cc/Praxi-study/Praxi-Pipeline/get_layer_changes/"
# homed = "/pipelines/component/"
src = homed+"src/"
if not Path(src).exists():
    Path(src).mkdir()
    # os.chmod(src, 777)
cwd = homed+"cwd/"
if not Path(cwd).exists():
    Path(cwd).mkdir()
    # os.chmod(cwd, 777)

# # LOKI_TOKEN=$(oc whoami -t)
# # curl -H "Authorization: Bearer $LOKI_TOKEN" "https://grafana-open-cluster-management-observability.apps.nerc-ocp-infra.rc.fas.harvard.edu/api/datasources/proxy/1/api/v1/query" --data-urlencode 'query=kube_pod_container_info{namespace="ai4cloudops-f7f10d9"}' | jq

# grafana_addr = 'https://grafana-open-cluster-management-observability.apps.nerc-ocp-infra.rc.fas.harvard.edu/api/datasources/proxy/1/api/v1/query'

# headers={
#     'Authorization': 'Bearer sha256~1GInrC-5iKWU-HpwVLRmAzefAm64vsgEp3wNewZPNBw',
#     'Content-Type': 'application/x-www-form-urlencoded'
#     }

# name_space = "ai4cloudops-f7f10d9"
# params = {
#     "query": "kube_pod_container_info{namespace='"+name_space+"'}"
#     }

# kube_pod_container_info = requests.get(grafana_addr, params=params, headers=headers)
# image_name = "/".join(kube_pod_container_info.json()['data']['result'][0]['metric']['image'].split("/")[1:])

# image_name = "zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0"
# image_name = "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4"
# image_name = "zongshun96/introspected_container:0.01"
image_name_l = ["zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0"]
changesets_d = defaultdict(list)

for image_name in image_name_l:
    image_d = {}
    image_meta_d = {}
    image_layer_dir = cwd+image_name.replace('/','_')+"/"
    image_downloader.download_image(repository=image_name, tag="latest", output_dir=image_layer_dir)

    with open(cwd+f"logfile_reading_tar_{image_name.replace('/','_')}.log", "w") as log_file:
        for root, subdirs, files in os.walk(image_layer_dir):
            for file_name in files:
                print(os.path.join(root, file_name))
                # print(file_name)
                if file_name == f"manifest_{image_name.replace('/','_')}.json":
                    # json_file = tar.extractfile(member)
                    with open(os.path.join(root, file_name), "r") as json_file:
                        content = json.load(json_file)
                        image_meta_d[file_name] = content
                        pprint(file_name, log_file)
                        pprint(content, log_file)
                        pprint("\n", log_file)
                elif file_name[-6:] == "tar.gz":
                    # tar_bytes = io.BytesIO(tar.extractfile(member).read())
                    tar_file = os.path.join(root, file_name)
                    inner_tar = tarfile.open(tar_file)
                    image_d[file_name] = inner_tar.getnames()
                    pprint(tar_file, log_file)
                    pprint(inner_tar.getnames(), log_file)
                    pprint("\n", log_file)
                    inner_tar.close()
    # shutil.rmtree(image_layer_dir)
                

    changesets_dir = image_layer_dir+"changesets/"
    if not Path(changesets_dir).exists():
        Path(changesets_dir).mkdir(parents=True)
        # os.chmod(changesets_dir, 777)
    # with open(cwd+f"logfile_changeset_gen_{image_name.replace('/','_')}.log", "w") as log_file:
    for layer in image_meta_d[f"manifest_{image_name.replace('/','_')}.json"]["layers"]:
        # yaml_in = {'open_time': open_time, 'close_time': close_time, 'label': label, 'changes': changes}
        yaml_in = {'labels': ['unknown'], 'changes': image_d[f"{layer['digest'].replace(':', '_')}.tar.gz"]}
        changeset_filename = get_free_filename("unknown", changesets_dir, ".yaml")
        with open(changeset_filename, 'w') as outfile:
            print("gen_changeset", os.path.dirname(outfile.name))
            print("gen_changeset", changeset_filename)
            yaml.dump(yaml_in, outfile, default_flow_style=False)
        changesets_d[image_name].append(yaml_in)

cs_dump_path = "/home/cc/Praxi-study/Praxi-Pipeline/get_layer_changes/cwd/changesets_d_dump"
# cs_path = "/home/cc/Praxi-study/Praxi-Pipeline/get_layer_changes/cwd/unknown"
with open(cs_dump_path, 'wb') as writer:
    pickle.dump(changesets_d, writer)