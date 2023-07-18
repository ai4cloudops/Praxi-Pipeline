import tarfile, sys, io, json, os, tempfile, subprocess, yaml
import requests
from pathlib import Path
from pprint import pprint

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

def run():
    # homed = "/home/ubuntu/Praxi-Pipeline/get_layer_changes/"
    homed = "/pipelines/component/"
    src = homed+"src/"
    # if not Path(src).exists():
    #     Path(src).mkdir()
    #     os.chmod(src, 777)
    cwd = homed+"cwd/"
    # if not Path(cwd).exists():
    #     Path(cwd).mkdir()
    #     os.chmod(cwd, 777)

    # LOKI_TOKEN=$(oc whoami -t)
    # curl -H "Authorization: Bearer $LOKI_TOKEN" "https://grafana-open-cluster-management-observability.apps.nerc-ocp-infra.rc.fas.harvard.edu/api/datasources/proxy/1/api/v1/query" --data-urlencode 'query=kube_pod_container_info{namespace="ai4cloudops-f7f10d9"}' | jq

    grafana_addr = 'https://grafana-open-cluster-management-observability.apps.nerc-ocp-infra.rc.fas.harvard.edu/api/datasources/proxy/1/api/v1/query'

    headers={
        'Authorization': 'Bearer sha256~6qRDv08phFx7mpcxDmVS14do1KsKUXVbFo0Olu5k0kQ',
        'Content-Type': 'application/x-www-form-urlencoded'
        }

    name_space = "ai4cloudops-f7f10d9"
    params = {
        "query": "kube_pod_container_info{namespace='"+name_space+"'}"
        }

    kube_pod_container_info = requests.get(grafana_addr, params=params, headers=headers)
    image_name = "/".join(kube_pod_container_info.json()['data']['result'][0]['metric']['image'].split("/")[1:])


    cmd1 = "bash "+src+"download-frozen-image-v2.sh "+cwd+"introspected_container "+image_name
    p_cmd1 = subprocess.Popen(cmd1.split(" "), stdin=subprocess.PIPE)
    p_cmd1.communicate()

    image_layer_dir = cwd+"introspected_container"



    image_d = {}
    image_meta_d = {}
    with open(cwd+"logfile_reading_tar_introspected.log", "w") as log_file:
        for root, subdirs, files in os.walk(image_layer_dir):
            for file_name in files:
                print(os.path.join(root, file_name))
                # print(file_name)
                if file_name == "manifest.json":
                    # json_file = tar.extractfile(member)
                    with open(os.path.join(root, file_name), "r") as json_file:
                        content = json.load(json_file)
                        image_meta_d[file_name] = content
                        pprint(file_name, log_file)
                        pprint(content, log_file)
                        pprint("\n", log_file)
                elif file_name == "json":
                    # json_file = tar.extractfile(member)
                    with open(os.path.join(root, file_name), "r") as json_file:
                        content = json.load(json_file)
                        image_meta_d[root] = content
                        pprint(root, log_file)
                        pprint(content, log_file)
                        pprint("\n", log_file)
                elif file_name[-4:] == "json":
                    # json_file = tar.extractfile(member)
                    with open(os.path.join(root, file_name), "r") as json_file:
                        content = json.load(json_file)
                        image_meta_d[file_name] = content
                        pprint(file_name, log_file)
                        pprint(content, log_file)
                        pprint("\n", log_file)
                elif file_name[-3:] == "tar":
                    # tar_bytes = io.BytesIO(tar.extractfile(member).read())
                    tar_file = os.path.join(root, file_name)
                    inner_tar = tarfile.open(tar_file)
                    image_d[root.split("/")[-1]] = inner_tar.getnames()
                    pprint(tar_file, log_file)
                    pprint(inner_tar.getnames(), log_file)
                    pprint("\n", log_file)
                    inner_tar.close()



    changesets_l = []             
    changesets_dir = cwd+"changesets/"
    # if not Path(changesets_dir).exists():
    #     Path(changesets_dir).mkdir()
    #     os.chmod(changesets_dir, 777)
    with open(cwd+"logfile_changeset_gen_introspected.log", "w") as log_file:
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
                pprint(layer, log_file)
                pprint(image_config_history['created_by'], log_file)
                pprint(image_config_history['created'], log_file)
                pprint(image_d[layer.split("/")[0]], log_file)
                # print(image_config_history)

                # yaml_in = {'open_time': open_time, 'close_time': close_time, 'label': label, 'changes': changes}
                yaml_in = {'labels': ['unknown'], 'changes': image_d[layer.split("/")[0]]}
                changeset_filename = get_free_filename("unknown", changesets_dir, ".yaml")
                with open(changeset_filename, 'w') as outfile:
                    print("gen_changeset", os.path.dirname(outfile.name))
                    print("gen_changeset", changeset_filename)
                    yaml.dump(yaml_in, outfile, default_flow_style=False)
                changesets_l.append(yaml_in)
                
    return changesets_l

if __name__ == "__main__":
    run()