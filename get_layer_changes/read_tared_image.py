import tarfile, sys, io, json, os, tempfile
import requests, docker
from pprint import pprint


client = docker.DockerClient(base_url='unix://var/run/docker.sock')

# kube_pod_container_info{namespace="default"}
prothemus_server_addr = 'localhost'
name_space = "default"
kube_pod_container_info = requests.get("http://"+prothemus_server_addr+":9090/api/v1/query?query=kube_pod_container_info{namespace='"+name_space+"'}&start=1687801230.165&end=1687804830.165&step=14")
image_name = kube_pod_container_info.json()['data']['result'][0]['metric']['image']

image = client.images.pull(image_name)
# image = client.images.get("busybox:latest")
tmp = tempfile.NamedTemporaryFile()
with open(tmp.name, 'wb') as f:
    for chunk in image.save():
        f.write(chunk)

with open('/home/cc/Praxi-study/vw-kubeflow-pipeline/Praxi-Pipeline/get_layer_changes/image/image.tar', 'wb') as f:
    for chunk in image.save():
        f.write(chunk)

image_d = {}
image_meta_d = {}
with open("logfile_reading_tar_introspected.log", "w") as log_file:

    # tar = tarfile.open('/home/cc/Praxi-study/vw-kubeflow-pipeline/Praxi-Pipeline/get_layer_changes/image/image.tar')
    tar = tarfile.open(tmp.name)
    # pprint(tar.getnames())
    for member in tar.getmembers():
        # pprint(member.name)
        if member.isfile():
            # dir_path = member.name.split("/")[0]
            # file_name = member.name.split("/")[-1]
            dir_path = os.path.split(member.name)[0]
            file_name = os.path.split(member.name)[1]
            if file_name[-3:] == "tar":
                tar_bytes = io.BytesIO(tar.extractfile(member).read())
                inner_tar = tarfile.open(fileobj=tar_bytes)
                image_d[member.name] = inner_tar.getnames()
                pprint(member.name, log_file)
                pprint(inner_tar.getnames(), log_file)
                pprint("\n", log_file)
            elif file_name == "manifest.json":
                json_file = tar.extractfile(member)
                content = json.load(json_file)
                image_meta_d[file_name] = content
                pprint(file_name, log_file)
                pprint(content, log_file)
                pprint("\n", log_file)
            elif file_name == "json":
                json_file = tar.extractfile(member)
                content = json.load(json_file)
                image_meta_d[dir_path] = content
                pprint(dir_path, log_file)
                pprint(content, log_file)
                pprint("\n", log_file)
            elif file_name[-4:] == "json":
                json_file = tar.extractfile(member)
                content = json.load(json_file)
                image_meta_d[file_name] = content
                pprint(file_name, log_file)
                pprint(content, log_file)
                pprint("\n", log_file)
    tar.close()

with open("logfile_changeset_gen_introspected.log", "w") as log_file:
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
            pprint(image_d[layer], log_file)
            # print(image_config_history)

    # yaml_in = {'open_time': open_time, 'close_time': close_time, 'label': label, 'changes': changes}