kfp_endpoint='http://localhost:8080'

from typing import NamedTuple

import kfp
import kfp.dsl as dsl
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op

def load_model(modfile_path: OutputPath(str), model_path: OutputPath(str)):
    '''Loads the vw model file and Hybrid class object'''
    import boto3
    import os
    modfile = '/pipelines/component/src/praxi-model.vw'
    s3 = boto3.resource(service_name='s3', 
                        region_name='', 
                        aws_access_key_id="", 
                        aws_secret_access_key="",)
    s3.Bucket('praxi-model').download_file(Key='new-iter.vw', Filename=modfile)
    os.popen('cp {0} {1}'.format(modfile, modfile_path))

    model = '/pipelines/component/src/mod_file.p'
    s3.Bucket('praxi-model').download_file(Key='new-iter.p', Filename=model)
    os.popen('cp {0} {1}'.format(model, model_path))
    print(model)
generate_loadmod_op = kfp.components.create_component_from_func(load_model, output_component_file='generate_loadmod_op.yaml', base_image="lkorver/praxi-vw-base:0.346")



def get_tags(user_in: input(str), tags_path: OutputPath(str), output_args_path: OutputPath(str)):
    import tagset_gen
    import yaml
    from yaml import Loader
    import os
    import json
    import pickle
    import cp_tagsets

    tags = []

    count = 0
    target_dir = '/pipelines/component/src/ml_70'
    for tag_file in os.listdir(target_dir):
        if(tag_file[-3:] == 'tag'):
            with open(target_dir + '/' + tag_file, 'rb') as tf:
                tag = yaml.load(tf, Loader = Loader)
                tags.append(tag)
            count += 1
            if (count > 500):
                break
        
    with open(tags_path, 'wb') as writer:
        pickle.dump(tags, writer)
    print(len(tags))

    with open(output_args_path, 'wb') as argf:
        pickle.dump("multilabel", argf)
        
get_tagset_op = kfp.components.create_component_from_func(get_tags, output_component_file='get_tagset_component.yaml', base_image="lkorver/praxi-columbus-base:0.135")


def gen_prediction( modfile_path: InputPath(str),model_path: InputPath(str), test_tags_path: InputPath(str), prediction_path: OutputPath(str)):
    '''generate prediction given model'''
    import main
    import os
    import json
    import pickle
    import time
    import boto3
    from hybrid_tags import Hybrid
    args = main.get_inputs()
    data_loaded = []
    
    model = pickle.load(open(model_path, "rb"))                                 
    model.vw_modelfile = modfile_path

    with open(test_tags_path, 'rb') as reader:
        data_loaded = pickle.load(reader)

    # with open(created_tags_path, 'r') as stream:
    #     for line in stream:
    #         temp = json.loads(line)
    #         if (type(temp) != None):
    #             data_loaded.append(temp)

    print("labs",model.all_labels)
    preds = main.get_preds(model, data_loaded, args)
    print("output", preds)
    with open(prediction_path, 'wb') as writer:
        pickle.dump(preds, writer) 

    time.sleep(5000)
gen_prediction_op = kfp.components.create_component_from_func(gen_prediction, output_component_file='generate_pred_component.yaml', base_image="lkorver/praxi-vw-base:0.428") 



def praxi_pipeline():

    tags = get_tagset_op("")
    model = generate_loadmod_op()
    model.execution_options.caching_strategy.max_cache_staleness = "P0D"

    prediction = gen_prediction_op(model.outputs["modfile"],model.outputs["model"], tags.outputs["tags"])
    

kfp.Client(host=kfp_endpoint).create_run_from_pipeline_func(praxi_pipeline, arguments={})