

kubeflow_endpoint="https://praxi-xgb-incremental-kfp-endpoint-praxi-xgb-incremental.apps.nerc-ocp-test.rc.fas.harvard.edu"
bearer_token = "sha256~wwkawUJv8w3WqJlcoDfaotsiZQjpRtiVX5pdZoqcuMM" # oc whoami --show-token

from typing import NamedTuple

import os
import kfp, kfp_tekton, kubernetes
import kfp.dsl as dsl
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op

os.environ["DEFAULT_STORAGE_CLASS"] = "ocs-external-storagecluster-ceph-rbd"
os.environ["DEFAULT_ACCESSMODES"] = "ReadWriteOnce"

def load_model(clf_path: OutputPath(str), index_tag_mapping_path: OutputPath(str), tag_index_mapping_path: OutputPath(str), index_label_mapping_path: OutputPath(str), label_index_mapping_path: OutputPath(str)):
    '''Loads the vw model file and Hybrid class object '''
    import boto3
    import os
    import time
    # time.sleep(50000)
    
    s3 = boto3.resource(service_name='s3', 
                        region_name='us-east-1', 
                        aws_access_key_id="AKIAXECNQISLAUNL67HV", 
                        aws_secret_access_key="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",)
    
    model_localpath = '/pipelines/component/src/model.json'
    index_tag_mapping_localpath = '/pipelines/component/src/index_tag_mapping'
    tag_index_mapping_localpath = '/pipelines/component/src/tag_index_mapping'
    index_label_mapping_localpath = '/pipelines/component/src/index_label_mapping'
    label_index_mapping_localpath = '/pipelines/component/src/label_index_mapping'

    s3.Bucket('praxi-model-xgb-02').download_file(Key='model.json', Filename=clf_path)
    # os.popen('cp {0} {1}'.format(model_localpath, clf_path))
    s3.Bucket('praxi-model-xgb-02').download_file(Key='index_tag_mapping', Filename=index_tag_mapping_path)
    # os.popen('cp {0} {1}'.format(index_tag_mapping_localpath, index_tag_mapping_path))
    s3.Bucket('praxi-model-xgb-02').download_file(Key='tag_index_mapping', Filename=tag_index_mapping_path)
    # os.popen('cp {0} {1}'.format(tag_index_mapping_localpath, tag_index_mapping_path))
    s3.Bucket('praxi-model-xgb-02').download_file(Key='index_label_mapping', Filename=index_label_mapping_path)
    # os.popen('cp {0} {1}'.format(index_label_mapping_localpath, index_label_mapping_path))
    s3.Bucket('praxi-model-xgb-02').download_file(Key='label_index_mapping', Filename=label_index_mapping_path)
    # os.popen('cp {0} {1}'.format(label_index_mapping_localpath, label_index_mapping_path))

    # s3.Bucket('praxi-model-xgb-02').download_file(Key='model.json', Filename=model_localpath)
    # os.popen('cp {0} {1}'.format(model_localpath, clf_path))
    # s3.Bucket('praxi-model-xgb-02').download_file(Key='index_tag_mapping', Filename=index_tag_mapping_localpath)
    # os.popen('cp {0} {1}'.format(index_tag_mapping_localpath, index_tag_mapping_path))
    # s3.Bucket('praxi-model-xgb-02').download_file(Key='tag_index_mapping', Filename=tag_index_mapping_localpath)
    # os.popen('cp {0} {1}'.format(tag_index_mapping_localpath, tag_index_mapping_path))
    # s3.Bucket('praxi-model-xgb-02').download_file(Key='index_label_mapping', Filename=index_label_mapping_localpath)
    # os.popen('cp {0} {1}'.format(index_label_mapping_localpath, index_label_mapping_path))
    # s3.Bucket('praxi-model-xgb-02').download_file(Key='label_index_mapping', Filename=label_index_mapping_localpath)
    # os.popen('cp {0} {1}'.format(label_index_mapping_localpath, label_index_mapping_path))
    # # time.sleep(50000)

generate_loadmod_op = kfp.components.create_component_from_func(load_model, output_component_file='generate_loadmod_op.yaml', base_image="zongshun96/load_model_s3:0.01")


def generate_changesets(user_in: str, cs_path: OutputPath(str), args_path: OutputPath(str)):
    import read_layered_image
    import pickle
    import time
    import yaml
    import boto3
    # import os
    # import json
    s3 = boto3.resource(service_name='s3', 
                        region_name='us-east-1', 
                        aws_access_key_id="AKIAXECNQISLAUNL67HV", 
                        aws_secret_access_key="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",)
    
    changesets_l = read_layered_image.run()
    # time.sleep(5000)
    # debug
    for ind, changeset in enumerate(changesets_l):
        with open("/pipelines/component/cwd/changesets/changesets_l"+str(ind)+".yaml", 'w') as writer:
            # yaml.dump(changesets_l, writer)
            yaml.dump(changeset, writer, default_flow_style=False)
        s3.Bucket('praxi-interm-1').upload_file("/pipelines/component/cwd/changesets/changesets_l"+str(ind)+".yaml", "changesets_l"+str(ind)+".yaml")
    # pass data to next component
    with open(cs_path, 'wb') as writer:
        pickle.dump(changesets_l, writer)
    with open(args_path, 'wb') as argfile:
        pickle.dump(user_in, argfile)
    # time.sleep(5000)
generate_changeset_op = kfp.components.create_component_from_func(generate_changesets, output_component_file='generate_changeset_component.yaml', base_image="zongshun96/prom-get-layers:0.02")

def generate_tagset(input_args_path: InputPath(str), changeset_path: InputPath(str), output_text_path: OutputPath(str), output_args_path: OutputPath(str)):
    '''generate tagset from the changeset'''
    # import tagset_gen
    from columbus.columbus import columbus
    import json
    import pickle
    import os
    import time
    import boto3
    # from function import changeset_gen
    s3 = boto3.resource(service_name='s3', 
                        region_name='us-east-1', 
                        aws_access_key_id="AKIAXECNQISLAUNL67HV", 
                        aws_secret_access_key="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",)

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
    with open("/pipelines/component/cwd/changesets_l_dump", 'w') as writer:
        for change_dict in changesets_l:
            writer.write(json.dumps(change_dict) + '\n')
    for ind, tag_dict in enumerate(tagsets_l):
        with open("/pipelines/component/cwd/tagsets_"+str(ind)+".tag", 'w') as writer:
            writer.write(json.dumps(tag_dict) + '\n')
        s3.Bucket('praxi-interm-1').upload_file("/pipelines/component/cwd/tagsets_"+str(ind)+".tag", "tagsets_"+str(ind)+".tag")
    # time.sleep(5000)

    # Pass data to next component
    # for ind, tag_dict in enumerate(tagsets_l):
    #     with open(output_text_path+"/tagsets_"+str(ind)+".tag", 'w') as writer:
    #         writer.write(json.dumps(tag_dict) + '\n')
    with open(output_text_path, 'wb') as writer:
        # for tag_dict in tag_dict_gen:
        #     writer.write(json.dumps(tag_dict) + '\n')
        pickle.dump(tagsets_l, writer)
    with open(output_args_path, 'wb') as argfile:
        pickle.dump(user_in, argfile)
generate_tagset_op = kfp.components.create_component_from_func(generate_tagset, output_component_file='generate_tagset_component.yaml', base_image="zongshun96/taggen_openshift:0.01")


def gen_prediction(clf_path: InputPath(str), index_tag_mapping_path: InputPath(str), tag_index_mapping_path: InputPath(str), index_label_mapping_path: InputPath(str), label_index_mapping_path: InputPath(str), test_tags_path: InputPath(str), prediction_path: OutputPath(str)):
# def gen_prediction(model_path: InputPath(str), modfile_path: InputPath(str), test_tags_path: InputPath(str), created_tags_path: InputPath(str), prediction_path: OutputPath(str)):
    '''generate prediction given model'''
    # import main
    import os
    import yaml
    import pickle
    import time
    import tagsets_XGBoost
    import xgboost as xgb
    import boto3
    # time.sleep(5000)

    # args = main.get_inputs()
    s3 = boto3.resource(service_name='s3', 
                        region_name='us-east-1', 
                        aws_access_key_id="AKIAXECNQISLAUNL67HV", 
                        aws_secret_access_key="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",)
    cwd = "/pipelines/component/cwd/"
    # cwd = "/home/ubuntu/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd/"

    # # load from previous component
    # with open(test_tags_path, 'rb') as reader:
    #     tagsets_l = pickle.load(reader)
    tagset_files, feature_matrix, label_matrix = tagsets_XGBoost.tagsets_to_matrix(test_tags_path, index_tag_mapping_path, tag_index_mapping_path, index_label_mapping_path, label_index_mapping_path, train_flag=False, cwd=cwd)
    BOW_XGB = xgb.XGBClassifier(max_depth=10, learning_rate=0.1,silent=False, objective='binary:logistic', \
                      booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                      subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1)
    BOW_XGB.load_model(clf_path)


    # # debug
    # with open("/pipelines/component/cwd/tagsets.log", 'w') as writer:
    #     for tag_dict in tagsets_l:
    #         writer.write(json.dumps(tag_dict) + '\n')
    # time.sleep(5000)
    # print("labs",clf.all_labels)

    # prediction
    pred_label_matrix = BOW_XGB.predict(feature_matrix)
    results = tagsets_XGBoost.one_hot_to_names(index_label_mapping_path, pred_label_matrix)
    # print("output", results)

    # # debug
    # with open("/pipelines/component/cwd/summary.log", 'w') as writer:
    #     main.print_multilabel_results(results, writer, args=clf.get_args())
    # with open(index_label_mapping_path, 'rb') as fp:
    #     labels = np.array(pickle.load(fp))
    # tagsets_XGBoost.print_metrics(cwd, 'metrics_iter.out', test_label_matrix_iter, pred_label_matrix_iter, labels)

    # Pass data to next component
    with open(prediction_path, 'wb') as writer:
        pickle.dump(results, writer) 
    with open(cwd+"pred_l_dump", 'w') as writer:
        # for pred in results:
        for pred in results.values():
            writer.write(f"{pred}\n")
    with open(cwd+"pred_d_dump", 'w') as writer:
        results_d = {}
        for k,v in results.items():
            results_d[int(k)] = v
        yaml.dump(results_d, writer)
    s3.Bucket('praxi-interm-1').upload_file(cwd+"pred_l_dump", "pred_l_dump")
    s3.Bucket('praxi-interm-1').upload_file(cwd+"pred_d_dump", "pred_d_dump")

    # debug
    # time.sleep(5000)
gen_prediction_op = kfp.components.create_component_from_func(gen_prediction, output_component_file='generate_pred_component.yaml', base_image="zongshun96/prediction_xgb_openshift:0.01") 


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


    # Pipeline design
    model = generate_loadmod_op().apply(use_image_pull_policy()).add_affinity(affinity)
    change_test = generate_changeset_op("test").apply(use_image_pull_policy()).add_affinity(affinity)
    tag_test = generate_tagset_op(change_test.outputs["args"], change_test.outputs["cs"]).apply(use_image_pull_policy()).add_affinity(affinity)
    prediction = gen_prediction_op(model.outputs["clf"],model.outputs["index_tag_mapping"],model.outputs["tag_index_mapping"],model.outputs["index_label_mapping"],model.outputs["label_index_mapping"], tag_test.outputs["output_text"]).apply(use_image_pull_policy()).add_affinity(affinity)

if __name__ == "__main__":

    client = kfp_tekton.TektonClient(
            host=kubeflow_endpoint,
            existing_token=bearer_token,
            ssl_ca_cert = '/home/ubuntu/cert/ca.crt'
        )
    # client = kfp.Client(host=kfp_endpoint)
    client.create_run_from_pipeline_func(praxi_pipeline, arguments={})
    # print(client.list_experiments())