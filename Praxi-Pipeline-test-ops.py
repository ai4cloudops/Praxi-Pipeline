

kubeflow_endpoint="https://praxi-kfp-endpoint-praxi.apps.nerc-ocp-test.rc.fas.harvard.edu"
bearer_token = "sha256~CsP0wgf1WK_fA2ekbvVhUjnUgD3UvusuI7BvT2bu1Xo" # oc whoami --show-token

from typing import NamedTuple

import os
import kfp, kfp_tekton, kubernetes
import kfp.dsl as dsl
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op

os.environ["DEFAULT_STORAGE_CLASS"] = "ocs-external-storagecluster-ceph-rbd"
os.environ["DEFAULT_ACCESSMODES"] = "ReadWriteOnce"

def load_model(vw_model_path: OutputPath(str), clf_path: OutputPath(str)):
    '''Loads the vw model file and Hybrid class object '''
    import boto3
    import os
    import time
    # time.sleep(5000)
    vw_model_localpath = '/pipelines/component/src/vw_model.vw'
    s3 = boto3.resource(service_name='s3', 
                        region_name='us-east-1', 
                        aws_access_key_id="AKIAXECNQISLO5332P6S", 
                        aws_secret_access_key="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",)
    s3.Bucket('praxi-model-1').download_file(Key='praxi-model.vw', Filename=vw_model_localpath)
    os.popen('cp {0} {1}'.format(vw_model_localpath, vw_model_path))

    clf_localpath = '/pipelines/component/src/clf.p'
    s3.Bucket('praxi-model-1').download_file(Key='praxi-model.p', Filename=clf_localpath)
    os.popen('cp {0} {1}'.format(clf_localpath, clf_path))
    # time.sleep(5000)

generate_loadmod_op = kfp.components.create_component_from_func(load_model, output_component_file='generate_loadmod_op.yaml', base_image="zongshun96/load_model_s3:0.01")


def generate_changesets(user_in: str, cs_path: OutputPath(str), args_path: OutputPath(str)):
    import read_layered_image
    import pickle
    import time
    import yaml
    # import os
    # import json
    
    changesets_l = read_layered_image.run()
    # time.sleep(5000)
    # debug
    for ind, changeset in enumerate(changesets_l):
        with open("/pipelines/component/cwd/changesets/changesets_l"+str(ind)+".yaml", 'w') as writer:
            # yaml.dump(changesets_l, writer)
            yaml.dump(changeset, writer, default_flow_style=False)
    # pass data to next component
    with open(cs_path, 'wb') as writer:
        pickle.dump(changesets_l, writer)
    with open(args_path, 'wb') as argfile:
        pickle.dump(user_in, argfile)
    # time.sleep(5000)
generate_changeset_op = kfp.components.create_component_from_func(generate_changesets, output_component_file='generate_changeset_component.yaml', base_image="zongshun96/prom-get-layers:0.01")

def generate_tagset(input_args_path: InputPath(str), changeset_path: InputPath(str), output_text_path: OutputPath(str), output_args_path: OutputPath(str)):
    '''generate tagset from the changeset'''
    # import tagset_gen
    from columbus.columbus import columbus
    import json
    import pickle
    import os
    import time
    # from function import changeset_gen

    # Load data from previous component
    with open(input_args_path, 'rb') as in_argfile:
        user_in = pickle.load(in_argfile)
    with open(changeset_path, 'rb') as in_changesets_l:
        changesets_l = pickle.load(in_changesets_l)
                              
    # Tagset Generator
    tagsets_l = []
    for changeset in changesets_l:
        # tags = tagset_gen.get_columbus_tags(changeset['changes'])
        tag_dict = columbus(changeset['changes'], freq_threshold=2)
        tags = ['{}:{}'.format(tag, freq) for tag, freq in tag_dict.items()]
        cur_dict = {'labels': changeset['labels'], 'tags': tags}
        tagsets_l.append(cur_dict)

    # Debug
    with open("/pipelines/component/cwd/changesets_logging", 'w') as writer:
        for change_dict in changesets_l:
            writer.write(json.dumps(change_dict) + '\n')
    with open("/pipelines/component/cwd/tagsets_logging", 'w') as writer:
        for tag_dict in tagsets_l:
            writer.write(json.dumps(tag_dict) + '\n')
    # time.sleep(5000)

    # Pass data to next component
    with open(output_text_path, 'wb') as writer:
        # for tag_dict in tag_dict_gen:
        #     writer.write(json.dumps(tag_dict) + '\n')
        pickle.dump(tagsets_l, writer)
    with open(output_args_path, 'wb') as argfile:
        pickle.dump(user_in, argfile)
generate_tagset_op = kfp.components.create_component_from_func(generate_tagset, output_component_file='generate_tagset_component.yaml', base_image="zongshun96/taggen_openshift:0.01")


def gen_prediction(clf_path: InputPath(str), vw_model_path: InputPath(str), test_tags_path: InputPath(str), prediction_path: OutputPath(str)):
# def gen_prediction(model_path: InputPath(str), modfile_path: InputPath(str), test_tags_path: InputPath(str), created_tags_path: InputPath(str), prediction_path: OutputPath(str)):
    '''generate prediction given model'''
    import main
    import os
    import json
    import pickle
    import time
    from hybrid_tags import Hybrid
    import boto3
    args = main.get_inputs()
    s3 = boto3.resource(service_name='s3', 
                        region_name='us-east-1', 
                        aws_access_key_id="AKIAXECNQISLO5332P6S", 
                        aws_secret_access_key="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",)

    # # load from previous component
    # data_loaded = []
    # with open(created_tags_path, 'r') as stream:
    #     for line in stream:
    #         temp = json.loads(line)
    #         if (type(temp) != None):
    #             data_loaded.append(temp)
    with open(test_tags_path, 'rb') as reader:
        tagsets_l = pickle.load(reader)
    with open(clf_path, 'rb') as reader:
        clf = pickle.load(reader)
    os.popen('cp {0} {1}'.format(vw_model_path, "/pipelines/component/cwd/vw_model.vw"))
    clf.vw_modelfile = "/pipelines/component/cwd/vw_model.vw"
    clf.use_temp_files = False
    clf.outdir = "/pipelines/component/cwd/"

    # debug
    with open("/pipelines/component/cwd/tagsets.log", 'w') as writer:
        for tag_dict in tagsets_l:
            writer.write(json.dumps(tag_dict) + '\n')
    # time.sleep(5000)
    # print("labs",clf.all_labels)

    # prediction
    results = main.predict(clf, tagsets_l)
    # print("output", results)

    # # debug
    # with open("/pipelines/component/cwd/summary.log", 'w') as writer:
    #     main.print_multilabel_results(results, writer, args=clf.get_args())

    # Pass data to next component
    with open(prediction_path, 'wb') as writer:
        pickle.dump(results, writer) 
    with open("/pipelines/component/cwd/pred_l_dump", 'w') as writer:
        # for pred in results:
        for pred in results:
            writer.write(f"{pred}\n")
    s3.Bucket('praxi-model-1').upload_file("/pipelines/component/cwd/pred_l_dump", "pred_l_dump")

    # debug
    # time.sleep(5000)
gen_prediction_op = kfp.components.create_component_from_func(gen_prediction, output_component_file='generate_pred_component.yaml', base_image="zongshun96/prediction_openshift:0.01") 


# # Reading bigger data
# @func_to_container_op
# def print_text(text_path: InputPath()): # The "text" input is untyped so that any data can be printed
#     '''Print text'''
#     with open(text_path, 'rb') as reader:
#         for line in reader:
#             print(line, end = '')
    
def add_node_selector(label_name: str, label_value: str, container_op: dsl.ContainerOp) -> None:
    container_op.add_node_selector_constraint(label_name=label_name, label_values=label_value)

def use_image_pull_policy(image_pull_policy='Always'):
    def _use_image_pull_policy(task):
        task.container.set_image_pull_policy(image_pull_policy)
        return task
    return _use_image_pull_policy
    

@kfp.dsl.pipeline(
    name="Submitted Pipeline",
)
def praxi_pipeline():
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
    terms = kubernetes.client.models.V1NodeSelectorTerm(
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


    model = generate_loadmod_op().apply(use_image_pull_policy()).add_affinity(affinity)
    change_test = generate_changeset_op("test").apply(use_image_pull_policy()).add_affinity(affinity)
    tag_test = generate_tagset_op(change_test.outputs["args"], change_test.outputs["cs"]).apply(use_image_pull_policy()).add_affinity(affinity)
    prediction = gen_prediction_op(model.outputs["clf"],model.outputs["vw_model"], tag_test.outputs["output_text"]).apply(use_image_pull_policy()).add_affinity(affinity)


if __name__ == "__main__":

    client = kfp_tekton.TektonClient(
            host=kubeflow_endpoint,
            existing_token=bearer_token,
            ssl_ca_cert = '/home/ubuntu/Praxi-Pipeline/cert/ca.crt'
        )
    # client = kfp.Client(host=kfp_endpoint)
    client.create_run_from_pipeline_func(praxi_pipeline, arguments={})
    # print(client.list_experiments())