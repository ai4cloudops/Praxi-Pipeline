kfp_endpoint='http://localhost:8080'

from typing import NamedTuple

import kfp
import kfp.dsl as dsl
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op

def load_model(modfile_path: OutputPath(str), model_path: OutputPath(str)):
    '''Loads the vw model file and Hybrid class object '''
    import boto3
    import os
    modfile = '/pipelines/component/src/praxi-model.vw'
    s3 = boto3.resource(service_name='s3', 
                        region_name='us-east-2', 
                        aws_access_key_id="AKIAUD42OAPRKHSSHZIR", 
                        aws_secret_access_key="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",)
    s3.Bucket('praxi-model').download_file(Key='praxi-model.vw', Filename=modfile)
    os.popen('cp {0} {1}'.format(modfile, modfile_path))

    model = '/pipelines/component/src/mod_file.p'
    s3.Bucket('praxi-model').download_file(Key='praxi-model.p', Filename=model)
    os.popen('cp {0} {1}'.format(model, model_path))

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

    target_dir = cp_tagsets.cp_tagsets()
    print(target_dir, os.listdir(target_dir))
    for tag_file in os.listdir(target_dir):
        if(tag_file[-3:] == 'tag'):
            with open(target_dir + '/' + tag_file, 'rb') as tf:
                tag = yaml.load(tf, Loader = Loader)
                tags.append(tag)
        
    with open(tags_path, 'wb') as writer:
        pickle.dump(tags, writer)
    print(len(tags))

    with open(output_args_path, 'wb') as argf:
        pickle.dump("multilabel", argf)
        
get_tagset_op = kfp.components.create_component_from_func(get_tags, output_component_file='get_tagset_component.yaml', base_image="lkorver/praxi-columbus-base:0.096")



# def gen_prediction(model_path: InputPath(str), modfile_path: InputPath(str), test_tags_path: InputPath(str), prediction_path: OutputPath(str)):
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
    
    model = pickle.load(open(model_path, "rb"))                                 ###
    model.vw_modelfile = modfile_path

    with open(test_tags_path, 'rb') as reader:
        data_loaded = pickle.load(reader)

    # with open(created_tags_path, 'r') as stream:
    #     for line in stream:
    #         temp = json.loads(line)
    #         if (type(temp) != None):
    #             data_loaded.append(temp)

    print("labs",model.all_labels)
    pred = main.test(model, data_loaded, args)
    print("output", pred)
    with open(prediction_path, 'wb') as writer:
        pickle.dump(pred, writer) 

    time.sleep(5000)
gen_prediction_op = kfp.components.create_component_from_func(gen_prediction, output_component_file='generate_pred_component.yaml', base_image="lkorver/praxi-vw-base:0.346") 



def praxi_pipeline():

    tags = get_tagset_op("")
    model = generate_loadmod_op()

    prediction = gen_prediction_op(model.outputs["modfile"],model.outputs["model"], tags.outputs["tags"])
    

kfp.Client(host=kfp_endpoint).create_run_from_pipeline_func(praxi_pipeline, arguments={})