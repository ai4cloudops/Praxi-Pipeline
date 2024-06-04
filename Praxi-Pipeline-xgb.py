

kubeflow_endpoint="https://ds-pipeline-pipelines-definition-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org"
bearer_token = "" # oc whoami --show-token

from typing import NamedTuple

import os
import kfp, kfp_tekton, kubernetes
import kfp.dsl as dsl
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op
from kubernetes import client as k8s_client

os.environ["DEFAULT_STORAGE_CLASS"] = "ocs-external-storagecluster-ceph-rbd"
os.environ["DEFAULT_ACCESSMODES"] = "ReadWriteOnce"

def load_model(clf_path: OutputPath(str)):
    '''Loads the vw model file and Hybrid class object '''
    import boto3
    import os
    import time
    # time.sleep(50000)
    
    s3 = boto3.resource(service_name='s3', 
                        region_name='us-east-1', 
                        aws_access_key_id="", 
                        aws_secret_access_key="",)

    s3.Bucket('praxi-model-xgb-02').download_file(Key='True25_1000submodel_verpak.zip', Filename=clf_path)

generate_loadmod_op = kfp.components.create_component_from_func(load_model, output_component_file='generate_loadmod_op.yaml', base_image="registry-route-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org/zongshun96/load_model_s3:0.01")


def generate_changesets(user_in: str, cs_path: OutputPath(str), args_path: OutputPath(str)):
    # import read_layered_image
    import pickle
    import time
    import yaml
    import boto3
    import tarfile, json, os
    import requests
    from pathlib import Path
    import shutil
    from pprint import pprint
    from collections import defaultdict
    # sys.path.insert(0, "/home/cc/Praxi-study/Praxi-Pipeline/get_layer_changes/src")
    import image_downloader
    s3 = boto3.resource(service_name='s3', 
                        region_name='us-east-1', 
                        aws_access_key_id="", 
                        aws_secret_access_key="",)
    # time.sleep(5000)

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


    # homed = "/home/cc/Praxi-study/Praxi-Pipeline/get_layer_changes/"
    homed = "/pipelines/component/"
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

    grafana_addr = 'https://grafana-open-cluster-management-observability.apps.nerc-ocp-infra.rc.fas.harvard.edu/api/datasources/proxy/1/api/v1/query'

    headers={
        'Authorization': 'Bearer sha256~1GInrC-5iKWU-HpwVLRmAzefAm64vsgEp3wNewZPNBw',
        'Content-Type': 'application/x-www-form-urlencoded'
        }

    name_space = "ai4cloudops-f7f10d9"
    params = {
        "query": "kube_pod_container_info{namespace='"+name_space+"'}"
        }

    kube_pod_container_info = requests.get(grafana_addr, params=params, headers=headers)
    # image_name = "/".join(kube_pod_container_info.json()['data']['result'][0]['metric']['image'].split("/")[1:])
    image_name_l = []
    for container_idx in range(len(kube_pod_container_info.json()['data']['result'])):
        image_name_l.append("/".join(kube_pod_container_info.json()['data']['result'][container_idx]['metric']['image'].split("/")[1:]))

    # image_name = "zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0"
    # image_name = "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4"
    # image_name = "zongshun96/introspected_container:0.01"
    # image_name_l = ["zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4","zongshun96/python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0", "zongshun96/python3_9-slim-bullseye.aws-lambda-powertools_v2_35_1_p_fiona_v1_9_4"]
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
        shutil.rmtree(image_layer_dir)
                    

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

    # cs_dump_path = "/home/cc/Praxi-study/Praxi-Pipeline/get_layer_changes/cwd/changesets_l_dump"
    # # cs_path = "/home/cc/Praxi-study/Praxi-Pipeline/get_layer_changes/cwd/unknown"
    # with open(cs_dump_path, 'wb') as writer:
    #     pickle.dump(changesets_l, writer)
    # # for idx, changeset in enumerate(changesets_l):
    # #     with open(cs_path+"-{%d}.yaml".format(idx), 'w') as writer:
    # #         yaml.dump(changeset, writer, default_flow_style=False)

    # time.sleep(5000)
    # debug
    print(changesets_d)
    with open("/pipelines/component/cwd/changesets/changesets_d.yaml", 'w') as writer:
        # yaml.dump(changesets_l, writer)
        yaml.dump(changesets_d, writer, default_flow_style=False)
    # s3.Bucket('praxi-interm-1').upload_file("/pipelines/component/cwd/changesets/changesets_l"+str(ind)+".yaml", "changesets_l"+str(ind)+".yaml")
    # pass data to next component
    with open(cs_path, 'wb') as writer:
        pickle.dump(changesets_d, writer)
    with open(args_path, 'wb') as argfile:
        pickle.dump(user_in, argfile)
generate_changeset_op = kfp.components.create_component_from_func(generate_changesets, output_component_file='generate_changeset_component.yaml', base_image="registry-route-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org/zongshun96/prom-get-layers:1.0")

def generate_tagset(input_args_path: InputPath(str), changeset_path: InputPath(str), output_text_path: OutputPath(str), output_args_path: OutputPath(str)):
    '''generate tagset from the changeset'''
    # import tagset_gen
    from columbus.columbus import columbus
    import json
    import pickle
    import os
    import time
    import boto3
    from collections import defaultdict
    # from function import changeset_gen
    s3 = boto3.resource(service_name='s3', 
                        region_name='us-east-1', 
                        aws_access_key_id="", 
                        aws_secret_access_key="",)

    # Load data from previous component
    with open(input_args_path, 'rb') as in_argfile:
        user_in = pickle.load(in_argfile)
    with open(changeset_path, 'rb') as in_changesets_d:
        changesets_d = pickle.load(in_changesets_d)
                              
    # Tagset Generator
    tagsets_d = defaultdict(list)
    for image_name, changeset_l_per_img in changesets_d.items():
        for changeset in changeset_l_per_img:
            # # tags = tagset_gen.get_columbus_tags(changeset['changes'])
            # tag_dict = columbus(changeset['changes'], freq_threshold=2)
            # tags = ['{}:{}'.format(tag, freq) for tag, freq in tag_dict.items()]
            # cur_dict = {'labels': changeset['labels'], 'tags': tags}
            tag_dict = columbus(changeset['changes'], freq_threshold=1)
            # tags = ['{}:{}'.format(tag, freq) for tag, freq in tag_dict.items()]
            cur_dict = {'labels': changeset['labels'], 'tags': tag_dict}
            tagsets_d[image_name].append(cur_dict)

    # Debug
    print(changesets_d)
    print("============================================")
    print(tagsets_d)
    with open("/pipelines/component/cwd/changesets_d_dump", 'w') as writer:
        json.dump(changesets_d, writer)
    with open("/pipelines/component/cwd/tagsets_d_dump", 'w') as writer:
        json.dump(tagsets_d, writer)
        # s3.Bucket('praxi-interm-1').upload_file("/pipelines/component/cwd/tagsets_"+str(ind)+".tag", "tagsets_"+str(ind)+".tag")

    # time.sleep(5000)
    # Pass data to next component
    # for ind, tag_dict in enumerate(tagsets_l):
    #     with open(output_text_path+"/tagsets_"+str(ind)+".tag", 'w') as writer:
    #         writer.write(json.dumps(tag_dict) + '\n')
    with open(output_text_path, 'wb') as writer:
        # for tag_dict in tag_dict_gen:
        #     writer.write(json.dumps(tag_dict) + '\n')
        pickle.dump(tagsets_d, writer)
    with open(output_args_path, 'wb') as argfile:
        pickle.dump(user_in, argfile)
generate_tagset_op = kfp.components.create_component_from_func(generate_tagset, output_component_file='generate_tagset_component.yaml', base_image="registry-route-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org/zongshun96/taggen_openshift:0.01")


def gen_prediction(user_in: str, clf_zip_path: InputPath(str), test_tags_path: InputPath(str), prediction_path: OutputPath(str)):
# def gen_prediction(model_path: InputPath(str), modfile_path: InputPath(str), test_tags_path: InputPath(str), created_tags_path: InputPath(str), prediction_path: OutputPath(str)):
    '''generate prediction given model'''
    # import main
    import zipfile
    import os, sys
    from pathlib import Path
    import yaml
    import pickle
    import time
    import tagsets_XGBoost_pickCVbatch_on_demand_expert
    import xgboost as xgb
    import boto3
    import numpy as np
    # import tqdm
    import multiprocessing as mp
    from collections import defaultdict
    # time.sleep(5000)


    op_durations = defaultdict(int)
    t_0 = time.time()

    # args = main.get_inputs()
    s3 = boto3.resource(service_name='s3', 
                        region_name='us-east-1', 
                        aws_access_key_id="", 
                        aws_secret_access_key="",)
    cwd = "/pipelines/component/cwd/"
    # cwd = "/home/ubuntu/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd/"


    t_clf_decompressing_0 = time.time()
    # Path to the zip file (include the full path if the file is not in the current directory)
    zip_file_path = clf_zip_path
    # Directory where you want to extract the files
    cwd_clf = cwd
    # Check if the extraction directory exists, if not, create it
    if not os.path.exists(cwd_clf):
        os.makedirs(cwd_clf)
    # Unzipping the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(cwd_clf)
    print(f'Files extracted to {cwd_clf}')
    t_clf_decompressing_t = time.time()
    op_durations[f"total_clf_decompressing_time"] = t_clf_decompressing_t-t_clf_decompressing_0


    dataset = "data_4"
    n_models = 1000
    shuffle_idx = 0
    test_sample_batch_idx = 0
    n_samples = 4
    n_jobs = 1
    clf_njobs = 32
    n_estimators = 100
    depth = 1
    input_size = None
    dim_compact_factor = 1
    tree_method = "exact"
    max_bin = 1
    with_filter = True
    freq = 25
    output_cwd = f"{cwd}/output/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak_on_demand_expert/"
    Path(output_cwd).mkdir(parents=True, exist_ok=True)


    # Data 
    # t_data_0 = time.time()
    # tag_files_l = [tag_file for tag_file in os.listdir(test_tags_path) if tag_file[-3:] == 'tag']
    # tag_files_l_of_l, step = [], len(tag_files_l)//mp.cpu_count()+1
    # for i in range(0, len(tag_files_l), step):
    #     tag_files_l_of_l.append(tag_files_l[i:i+step])
    # pool = mp.Pool(processes=1)
    # data_instance_d_l = [pool.apply_async(map_tagfilesl, args=(test_tags_path, tag_files_l, cwd, True, freq)) for tag_files_l in tqdm(tag_files_l_of_l)]
    # data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
    # pool.close()
    # pool.join()
    # all_tags_set, all_label_set = set(), set()
    # tags_by_instance_l, labels_by_instance_l = [], []
    # tagset_files = []
    # for data_instance_d in data_instance_d_l:
    #     if len(data_instance_d) == 5:
    #             tagset_files.extend(data_instance_d['tagset_files'])
    #             all_tags_set.update(data_instance_d['all_tags_set'])
    #             tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
    #             all_label_set.update(data_instance_d['all_label_set'])
    #             labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
    
    t_data_0 = time.time()
    all_tags_set, all_label_set = set(), set()
    tags_by_instance_l, labels_by_instance_l = [], []
    tagset_files = []
    with open(test_tags_path, 'rb') as reader:
        tagsets_d = pickle.load(reader)
        for image_name, tagsets_l in tagsets_d.items():
            for layer_idx, tagset in enumerate(tagsets_l):
                tagset_files.append(f"{image_name}_layer_idx_{layer_idx}")
                instance_feature_tags_d = defaultdict(int)
                # # feature 
                # for tag_vs_count in tagset['tags']:
                #     k,v = tag_vs_count.split(":")
                #     all_tags_set.add(k)
                #     instance_feature_tags_d[k] += int(v)
                for k, v in tagset['tags'].items():
                    all_tags_set.add(k)
                    instance_feature_tags_d[k] += int(v)
                tags_by_instance_l.append(instance_feature_tags_d)
                # label
                if 'labels' in tagset:
                    all_label_set.update(tagset['labels'])
                    labels_by_instance_l.append(tagset['labels'])
                else:
                    all_label_set.add(tagset['label'])
                    labels_by_instance_l.append([tagset['label']])
    t_data_t = time.time()
    op_durations["total_data_load_time"] = t_data_t-t_data_0
    # # debugging
    # with open(f"{output_cwd}/labels_by_instance_l.yaml", 'w') as writer:
    #     yaml.dump(labels_by_instance_l, writer)
    # with open(f"{output_cwd}/tags_by_instance_l.yaml", 'w') as writer:
    #     yaml.dump(tags_by_instance_l, writer)

    # time.sleep(5000)

    # Models
    t_clf_path_0 = time.time()
    clf_path_l = []
    for i in range(n_models):
        clf_pathname = f"{cwd_clf}/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(clf_njobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/model_init.json"
        if os.path.isfile(clf_pathname):
            clf_path_l.append(clf_pathname)
        else:
            print(f"clf is missing: {clf_pathname}")
            sys.exit(-1)
    t_clf_path_t = time.time()
    op_durations["total_clf_path_load_time"] = t_clf_path_t-t_clf_path_0
    # with open(f"{output_cwd}/clf_path_l.yaml", 'w') as writer:
    #     yaml.dump(clf_path_l, writer)
    # print(clf_path_l)
    
    # time.sleep(5000)

    # Make inference
    # label_matrix_list, pred_label_matrix_list, labels_list = [], [], []
    # values_l_, pos_x_l_, pos_y_l_ = [],[],[]
    predicted_labels_dict, true_labels_dict = defaultdict(list), defaultdict(list)
    for clf_idx, clf_path in enumerate(clf_path_l):
        t_per_clf_0 = time.time()

        with open(clf_path[:-15]+'index_label_mapping', 'rb') as fp:
            clf_labels_l = pickle.load(fp)
            # labels_list.append(np.array(clf_labels_l))


        BOW_XGB = xgb.XGBClassifier(max_depth=10, learning_rate=0.1,silent=False, objective='binary:logistic', \
                        booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                        subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1)
        BOW_XGB.load_model(clf_path)
        BOW_XGB.set_params(n_jobs=n_jobs)
        feature_importance = BOW_XGB.feature_importances_

        t_per_clf_loading_t = time.time()
        op_durations[f"clf{clf_idx}_load_time"] = t_per_clf_loading_t-t_per_clf_0
        op_durations["total_clf_load_time"] += t_per_clf_loading_t-t_per_clf_0

        # # label_matrix_list_per_clf, pred_label_matrix_list_per_clf = [],[]
        # pred_label_matrix_list_per_clf = []
        # step = len(tag_files_l)
        for batch_first_idx in range(0, 1):
            t_encoder_0 = time.time()
            tagset_files_used, feature_matrix, label_matrix, instance_row_idx_set, instance_row_count, encoder_op_durations = tagsets_XGBoost_pickCVbatch_on_demand_expert.tagsets_to_matrix(test_tags_path, cwd=clf_path[:-15], all_tags_set=all_tags_set,all_label_set=all_label_set,tags_by_instance_l=tags_by_instance_l,labels_by_instance_l=labels_by_instance_l,tagset_files=tagset_files, feature_importance=feature_importance)
            # values_l_.extend(values_l)
            # pos_x_l_.extend(pos_x_l)
            # pos_y_l_.extend(pos_y_l)
            t_encoder_t = time.time()
            op_durations[f"encoder{clf_idx}_op_durations"] = encoder_op_durations
            op_durations[f"encoder{clf_idx}_time"] += t_encoder_t-t_encoder_0
            op_durations["total_encoder_time"] += t_encoder_t-t_encoder_0
            t_inference_0 = time.time()
            if feature_matrix.size != 0:
                # prediction
                pred_label_matrix = BOW_XGB.predict(feature_matrix)
            t_inference_t = time.time()
            op_durations[f"inference{clf_idx}_time"] += t_inference_t-t_inference_0
            op_durations["total_inference_time"] += t_inference_t-t_inference_0
            # # !!!!!!!!!!!!!!!!!!!!!!!! fill zeros for samples without recogonizable features by this clf
            t_decoding_batch_0 = time.time()
            # all_label_len = len(clf_labels_l)
            # # new_label_matrix = np.vstack(np.zeros((instance_row_count, all_label_len)))
            # # for instance_row_idx, new_instance_row_idx in enumerate(list(instance_row_idx_set)):
            # #     new_label_matrix[new_instance_row_idx, :] = label_matrix[instance_row_idx, :]
            # # label_matrix = new_label_matrix

            # new_pred_label_matrix = np.vstack(np.zeros((instance_row_count, all_label_len)))
            # for instance_row_idx, new_instance_row_idx in enumerate(list(instance_row_idx_set)):
            #     new_pred_label_matrix[new_instance_row_idx, :] = pred_label_matrix[instance_row_idx, :]
            # pred_label_matrix = new_pred_label_matrix

            # # label_matrix_list_per_clf.append(label_matrix)
            # pred_label_matrix_list_per_clf.append(pred_label_matrix)

            if instance_row_idx_set:
                pred_label_name_d = tagsets_XGBoost_pickCVbatch_on_demand_expert.one_hot_to_names('index_label_mapping', pred_label_matrix, mapping=clf_labels_l)
                predicted_labels_dict = tagsets_XGBoost_pickCVbatch_on_demand_expert.merge_preds(predicted_labels_dict, pred_label_name_d, instance_row_idx_set)
                # print(0)
            # label_name_d = one_hot_to_names('index_label_mapping', label_matrix, mapping=clf_labels_l)
            # true_labels_dict = merge_preds(true_labels_dict, label_name_d)
                
            t_decoding_batch_t = time.time()
            op_durations[f"decoding{clf_idx}_time"] += t_decoding_batch_t-t_decoding_batch_0

        
        t_decoding_0 = time.time()
        # # label_matrix_list_per_clf = np.vstack(label_matrix_list_per_clf)
        # pred_label_matrix_list_per_clf = np.vstack(pred_label_matrix_list_per_clf)
        # results = merge_preds(results, one_hot_to_names(clf_path[:-15]+'index_label_mapping', pred_label_matrix_list_per_clf))
        # # label_matrix_list.append(label_matrix_list_per_clf)
        # # pred_label_matrix_list.append(pred_label_matrix_list_per_clf)
        t_decoding_t = time.time()
        op_durations[f"decoding{clf_idx}_time"] += t_decoding_t-t_decoding_0


        
        t_per_clf_t = time.time()
        op_durations[f"clf{clf_idx}_time"] = t_per_clf_t-t_per_clf_0
        op_durations["total_clf_time"] += t_per_clf_t-t_per_clf_0
        # print("clf"+str(clf_idx)+" pred done")
        
        
    t_t = time.time()
    # op_durations["len(values_l_)"] = len(values_l_)
    # op_durations["len(pos_x_l_)"] = len(pos_x_l_)
    # op_durations["len(pos_y_l_)"] = len(pos_y_l_)
    op_durations["total_time"] = t_t-t_0
    with open(f"{output_cwd}metrics.yaml", 'w') as writer:
        yaml.dump(op_durations, writer)
    # print(predicted_labels_dict)

    # Pass data to next component
    with open(prediction_path, 'wb') as writer:
        pickle.dump(predicted_labels_dict, writer) 
    with open(output_cwd+"pred_l_dump", 'w') as writer:
        for pred in predicted_labels_dict.values():
            writer.write(f"{pred}\n")
    with open(output_cwd+"pred_d_dump", 'w') as writer:
        results_d = {}
        for k,v in predicted_labels_dict.items():
            results_d[int(k)] = v
        yaml.dump(results_d, writer)
    with open(output_cwd+"tagset_files_dump", 'w') as writer:
        yaml.dump(tagset_files, writer)
    s3.Bucket('praxi-interm-1').upload_file(output_cwd+"pred_l_dump", f"pred_l_dump{user_in}")
    s3.Bucket('praxi-interm-1').upload_file(output_cwd+"pred_d_dump", f"pred_d_dump{user_in}")
    s3.Bucket('praxi-interm-1').upload_file(output_cwd+"metrics.yaml", f"metrics{user_in}.yaml")
    s3.Bucket('praxi-interm-1').upload_file(output_cwd+"tagset_files_dump", f"tagset_files_dump{user_in}.yaml")

    # debug
    # time.sleep(5000)
gen_prediction_op = kfp.components.create_component_from_func(gen_prediction, output_component_file='generate_pred_component.yaml', base_image="registry-route-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org/zongshun96/prediction_xgb_openshift:1.1") 


# # Reading bigger data
# @func_to_container_op
# def print_text(text_path: InputPath()): # The "text" input is untyped so that any data can be printed
#     '''Print text'''
#     with open(text_path, 'rb') as reader:
#         for line in reader:
#             print(line, end = '')
    
def add_node_selector(label_name: str, label_value: str, container_op: dsl.ContainerOp) -> None:
    container_op.add_node_selector_constraint(label_name=label_name, label_values=label_value)

def use_image_pull_secret(op):
    """Function to apply the imagePullSecrets to the pod spec."""
    from kubernetes.client.models import V1Pod, V1PodSpec, V1ObjectMeta

    pod_spec = V1Pod(spec=V1PodSpec(image_pull_secrets=[{"name": "my-registry-secret"}]))
    op.pod_spec = pod_spec
    return op

def use_image_pull_policy(image_pull_policy='Always'):
    def _use_image_pull_policy(task):
        # task.container.apply(use_image_pull_secret)
        task.container.set_image_pull_policy(image_pull_policy)
        return task
    return _use_image_pull_policy
    

@kfp.dsl.pipeline(
    name="Submitted Pipeline",
)
def praxi_pipeline(trial_idx):
    # vop = dsl.VolumeOp(
    #     name="interm-pvc",
    #     resource_name="interm-pvc",
    #     size="1Gi",
    #     modes=dsl.VOLUME_MODE_RWM,
    #     volume_name="pvc-75829191-2c57-4630-ae3b-191c4d4d372f",
    #     storage_class="manual",
    #     generate_unique_name=False,
    #     action='apply',
    #     set_owner_reference=True
    # )




    # kubernetes.config.load_kube_config()
    # api = kubernetes.client.AppsV1Api()

    # # read current state
    # deployment = api.read_namespaced_deployment(name='foo', namespace='bar')

    # check current state
    #print(deployment.spec.template.spec.affinity)

    # create affinity objects
    terms = kubernetes.client.models.V1NodeSelectorTerm(    # GPU nodes had permission issues, so we enforce to use other nodes. Use this code to set node selector.
        match_expressions=[
            {'key': 'kubernetes.io/hostname',
            'operator': 'NotIn',
            'values': ["wrk-10", "wrk-11"]}
        ]
    )
    node_selector = kubernetes.client.models.V1NodeSelector(node_selector_terms=[terms])
    node_affinity = kubernetes.client.models.V1NodeAffinity(
        required_during_scheduling_ignored_during_execution=node_selector
    )
    affinity = kubernetes.client.models.V1Affinity(node_affinity=node_affinity)


    dsl.get_pipeline_conf().set_image_pull_secrets([k8s_client.V1ObjectReference(name="my-registry-secret"),k8s_client.V1ObjectReference(name="regcred")])

    # Pipeline design
    model = generate_loadmod_op().apply(use_image_pull_policy()).add_affinity(affinity)
    change_test = generate_changeset_op(str(trial_idx)).apply(use_image_pull_policy()).add_affinity(affinity)
    change_test.set_cpu_limit('4')
    change_test.set_cpu_request('4')
    change_test.set_memory_limit('5120Mi')
    change_test.set_memory_request('5120Mi')
    tag_test = generate_tagset_op(change_test.outputs["args"], change_test.outputs["cs"]).apply(use_image_pull_policy()).add_affinity(affinity)
    tag_test.set_cpu_limit('4')
    tag_test.set_cpu_request('4')
    tag_test.set_memory_limit('5120Mi')
    tag_test.set_memory_request('5120Mi')
    prediction = gen_prediction_op(str(trial_idx), model.outputs["clf"], tag_test.outputs["output_text"]).apply(use_image_pull_policy()).add_affinity(affinity)
    prediction.set_cpu_limit('4')
    prediction.set_cpu_request('4')
    prediction.set_memory_limit('5120Mi')
    prediction.set_memory_request('5120Mi')

if __name__ == "__main__":

    import time

    client = kfp_tekton.TektonClient(
            host=kubeflow_endpoint,
            existing_token=bearer_token,
            # ssl_ca_cert = '/home/ubuntu/cert/ca.crt'
        )
    # client = kfp.Client(host=kfp_endpoint)

    for trial_idx in range(0, 3):
        arguments = {"trial_idx": trial_idx}
        client.create_run_from_pipeline_func(praxi_pipeline, arguments=arguments)
        # print(client.list_experiments())

        time.sleep(1000)