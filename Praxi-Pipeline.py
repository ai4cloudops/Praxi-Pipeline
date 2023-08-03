

kfp_endpoint='http://localhost:8080'

from typing import NamedTuple

import kfp
import kfp.dsl as dsl
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op

def generate_changesets(user_in: input(str), single: input(bool), multi: input(bool), cs_path: OutputPath(str), args_path: OutputPath(str)):
    '''
    Input: user_in- a string specifying the packages to create changesets for, single and multi- True or False values, whether to create single or 
    multilabel changesets or both
    Output: generated changesets, passed into cs_path
    '''
    from function import changeset_gen
    import pickle
    import os
    import json
    print(user_in)
    packages = []
    for package in user_in.split():
        packages.append(package)
    changesets = changeset_gen.run(packages, single, multi)
    print(changesets)
    os.popen('cp -r {0} {1}'.format("/changesets", cs_path))
    with open(args_path, 'wb') as argfile:
        pickle.dump(user_in, argfile)
    print(os.listdir("/changesets"))
generate_changeset_op = kfp.components.create_component_from_func(generate_changesets, output_component_file='generate_changeset_component.yaml', base_image="lkorver/praxi-columbus-base:0.097")


def load_changesets(train_ts_path: OutputPath(str)):
    '''Loads changesets from a specified direrctory (changedir) and creates tagsets for them, returning tagsets'''
    import tagset_gen
    import yaml
    from yaml import Loader
    import os
    import json
    import pickle
    import tagset_gen
    changedir = "/pipelines/component/src/new_data"
    os.listdir(changedir)
    tag_dict_gen = tagset_gen.run(changedir)

    with open(train_ts_path, 'w') as writer:
         for tag_dict in tag_dict_gen:
             writer.write(json.dumps(tag_dict) + '\n')
    
get_cs_op = kfp.components.create_component_from_func(load_changesets, output_component_file='get_cs_component.yaml', base_image="lkorver/praxi-columbus-base:0.104")



def generate_tagset(input_args_path: InputPath(str), changeset_path: InputPath(str), output_text_path: OutputPath(str), output_args_path: OutputPath(str)):
    '''generate tagsets from the changesets'''
    import tagset_gen
    import json
    import pickle
    import os
    from function import changeset_gen
    change_dir = changeset_path
    tag_dict_gen = tagset_gen.run(change_dir)

    with open(input_args_path, 'rb') as in_argfile:
        user_in = pickle.load(in_argfile)
    
    with open(output_text_path, 'w') as writer:
         for tag_dict in tag_dict_gen:
             writer.write(json.dumps(tag_dict) + '\n')
             
    with open(output_args_path, 'wb') as argfile:
        pickle.dump(user_in, argfile)
generate_tagset_op = kfp.components.create_component_from_func(generate_tagset, output_component_file='generate_tagset_component.yaml', base_image="lkorver/praxi-columbus-base:0.097")



def load_tags(train_tags_path: OutputPath(str), test_tags_path: OutputPath(str), output_args_path: OutputPath(str)):
    '''Loads premade tagsets from specified directory (target_dir) and returns separated train and test datasets'''
    import tagset_gen
    import yaml
    from yaml import Loader
    import os
    import json
    import pickle
    import cp_tagsets

    train_tags = []
    test_tags = []

    count = 0
    target_dir = '/pipelines/component/src/sl_70'    #contains 32 SL of each of 69 labels, 2208 total
    for tag_file in os.listdir(target_dir):
        if(tag_file[-3:] == 'tag'):
            with open(target_dir + '/' + tag_file, 'rb') as tf:
                if (count%32 < 2):
                    tag = yaml.load(tf, Loader = Loader)
                    test_tags.append(tag)
                else:
                    tag = yaml.load(tf, Loader = Loader)
                    # train_tags.append(tag)
                count += 1

    # count = 0
    # target_dir = '/pipelines/component/src/sl_10'    #contains 32 SL of each of 10 labels, 320 total
    # for tag_file in os.listdir(target_dir):
    #     if(tag_file[-3:] == 'tag'):
    #         with open(target_dir + '/' + tag_file, 'rb') as tf:
    #             # if (count%32 < 2):
    #             if (count%32 > 1) and (count%32 < 4):
    #             # if (count%32 > 3) and (count%32 < 6):
    #                 tag = yaml.load(tf, Loader = Loader)
    #                 test_tags.append(tag)
    #             else:
    #                 tag = yaml.load(tf, Loader = Loader)
    #                 train_tags.append(tag)
    #             count += 1
    
    print("sl train amount",len(train_tags))

    count = 0
    added = 0
    target_dir = '/pipelines/component/src/ml_70' #contains ~12000 total ML tags, 5 of each combo
    for tag_file in os.listdir(target_dir):
        if(tag_file[-3:] == 'tag'):
            with open(target_dir + '/' + tag_file, 'rb') as tf:
                if (count%5 < 2):
                    tag = yaml.load(tf, Loader = Loader)
                    test_tags.append(tag)
                else:
                    tag = yaml.load(tf, Loader = Loader)
                    # train_tags.append(tag)
                if (count%1000 == 0):
                    print("count", count)
                count += 1
    
    # count = 0
    # added = 0
    # target_dir = '/pipelines/component/src/ml_10' #contains 450 total ML tags, 10 of each combo
    # for tag_file in os.listdir(target_dir):
    #     if(tag_file[-3:] == 'tag'):
    #         with open(target_dir + '/' + tag_file, 'rb') as tf:
    #             # if (count%5 < 2):
    #             # if (count%5 > 1) and (count%5 < 4):
    #             if (count%5 > 3) or (count%5 == 0):
    #                 tag = yaml.load(tf, Loader = Loader)
    #                 test_tags.append(tag)
    #             else:
    #                 tag = yaml.load(tf, Loader = Loader)
    #                 train_tags.append(tag)
    #             count += 1

    print("total train amount", len(train_tags))
    print("total test amount",len(test_tags))
    
    with open(train_tags_path, 'wb') as writer:
        pickle.dump(train_tags, writer)
    print(len(train_tags))
    
    with open(test_tags_path, 'wb') as writer:
        pickle.dump(test_tags, writer)
    print(len(test_tags))

    with open(output_args_path, 'wb') as argf:
        pickle.dump("multilabel", argf)
        
get_tagset_op = kfp.components.create_component_from_func(load_tags, output_component_file='get_tagset_component.yaml', base_image="lkorver/praxi-columbus-base:0.142")



def get_train_type(train_type: input(str)) -> str:
    import tagset_gen
    import json
    import pickle
    print(train_type)
    return train_type
get_traintype_op = kfp.components.create_component_from_func(get_train_type, output_component_file='get_traintype_component.yaml', base_image="lkorver/praxi-columbus-base:0.065")



def iterative_training(tagset_path: InputPath(str), modfile_path: OutputPath(str), model_path: OutputPath(str)):
    ''''''
    import main
    import os
    import json
    import pickle
    import time
    args = main.get_inputs() 
    data_loaded = []

    with open(tagset_path, 'rb') as reader:
        data_loaded = pickle.load(reader)
    
    # with open(tagset_path, 'r') as stream:
    #     for line in stream:
    #         temp = json.loads(line)
    #         if (type(temp) != None):
    #             data_loaded.append(temp)
    
    model = main.iterative_train(data_loaded, args)
    modfile = model.vw_modelfile
    print("modfile",modfile)
    print(model)
    os.popen('cp {0} {1}'.format(modfile, modfile_path))
    with open(model_path, 'wb') as modelfile:
        pickle.dump(model, modelfile)    
            
    # with open(test_tags_path, 'wb') as testfile:
    #     pickle.dump(testdat, testfile)
generate_ittrain_op = kfp.components.create_component_from_func(iterative_training, output_component_file='generate_ittrain_component.yaml', base_image="lkorver/praxi-vw-base:0.434") 



def multilabel_training(tagset_path: InputPath(str), modfile_path: OutputPath(str), model_path: OutputPath(str)):
    ''''''
    import main
    import os
    import json
    import pickle
    import time

    args = main.get_inputs() 
    data_loaded = []

    with open(tagset_path, 'rb') as reader:
        data_loaded = pickle.load(reader)

    # count = 0
    # with open(created_tags_path, 'r') as stream:
    #     for line in stream:
    #         temp = json.loads(line)
    #         if (type(temp) != None) and (count%4 >= 3):
    #             data_loaded.append(temp)
    #         count += 1

    # print(len(data_loaded))
    
    model = main.multilabel_train(data_loaded, args)
    modfile = model.vw_modelfile
    os.popen('cp {0} {1}'.format(modfile, modfile_path))
    with open(model_path, 'wb') as modfile:
        pickle.dump(model, modfile)

    # with open(test_tags_path, 'wb') as testfile:
    #     pickle.dump(testdat, testfile)

generate_multilabel_op = kfp.components.create_component_from_func(multilabel_training, output_component_file='generate_multilabel_component.yaml', base_image="lkorver/praxi-vw-base:0.434")



def gen_prediction(model_path: InputPath(str), modfile_path: InputPath(str), test_tags_path: InputPath(str), prediction_path: OutputPath(str)):
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

    with open(test_tags_path, 'rb') as reader:
        data_loaded = pickle.load(reader)

    # with open(created_tags_path, 'r') as stream:
    #     for line in stream:
    #         temp = json.loads(line)
    #         if (type(temp) != None):
    #             data_loaded.append(temp)
    
    with open(model_path, 'rb') as reader:
        model = pickle.load(reader)
    model.vw_modelfile = modfile_path
    print("labs",model.all_labels)
    pred, model = main.test(model, data_loaded, args)
    print("output", pred)
    with open(prediction_path, 'wb') as writer:
        pickle.dump(pred, writer) 

    mod_file = 'mod_file.p'
    with open(mod_file, 'wb') as mod: 
        pickle.dump(model, mod)

    # #export file to AWS S3 bucket 
    # s3 = boto3.resource(service_name='s3', 
    #                     region_name='', 
    #                     aws_access_key_id="", 
    #                     aws_secret_access_key="",)
    # for bucket in s3.buckets.all():
    #     print("bucket",bucket.name)
    # s3.Bucket('praxi-model').upload_file(modfile_path,'new-iter.vw')    
    # s3.Bucket('praxi-model').upload_file(mod_file,'new-iter.p')
    time.sleep(5000)
gen_prediction_op = kfp.components.create_component_from_func(gen_prediction, output_component_file='generate_pred_component.yaml', base_image="lkorver/praxi-vw-base:0.434")



def xgb_prediction(model_path: InputPath(str), modfile_path: InputPath(str),train_tags_path: InputPath(str), test_tags_path: InputPath(str), prediction_path: OutputPath(str)):
    '''Trains XGBoost model using train_tags and returns metrics for test_tags'''
    from model_testing_scripts import tagsets_to_matrix
    import os
    import json
    import pickle
    import time
    import boto3
    import yaml
    import time
    from main import get_free_filename
    cwd = os.getcwd()
    
    with open(train_tags_path, 'rb') as reader:
        tr_data_loaded = pickle.load(reader)
    with open(test_tags_path, 'rb') as reader:
        ts_data_loaded = pickle.load(reader)

    tr_tags_dir = '/train_tags'
    ts_tags_dir = '/test_tags'

    os.mkdir(tr_tags_dir)
    os.mkdir(ts_tags_dir)

    count = 0
    for tag in tr_data_loaded:
        file = get_free_filename("tag", tr_tags_dir, '.tag')
        with open(file, 'wb') as tf:
            pickle.dump(tag, tf)
        #     count +=1
        # if count > 1000:
        #     break
    count = 0
    for tag in ts_data_loaded:
        file = get_free_filename("tag",ts_tags_dir, '.tag')
        # if (count%10 == 0):   
        with open(file, 'wb') as tf:
            pickle.dump(tag, tf)
        count +=1
            
    tagsets_to_matrix.run(tr_tags_dir, ts_tags_dir, cwd)
    time.sleep(5000)
gen_xgbpred_op = kfp.components.create_component_from_func(xgb_prediction, output_component_file='generate_xgbpred_component.yaml', base_image="lkorver/praxi-vw-base:0.396")


# Reading bigger data
@func_to_container_op
def print_text(text_path: InputPath()): # The "text" input is untyped so that any data can be printed
    '''Print text'''
    with open(text_path, 'rb') as reader:
        for line in reader:
            print(line, end = '')
    
def praxi_pipeline():
    # packages = ""  #packages to generate changesets for
    # changeset = generate_changeset_op(packages, single=True, multi=True)
    #changeset.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # tagset = generate_tagset_op(changeset.outputs["args"], changeset.outputs["cs"])
    # #tagset.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # new_tags = get_cs_op()

    tags = get_tagset_op()
    
    #arguments = get_traintype_op(tagset.outputs["output_args"])
    arguments = get_traintype_op("iterative")
    with dsl.Condition(arguments.output == 'multilabel'):
        mul_train = generate_multilabel_op(tags.outputs["train_tags"])
        # mul_train.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # prediction = gen_prediction_op(mul_train.outputs["model"], mul_train.outputs["modfile"], tag_test.outputs["output_text"])
        prediction = gen_prediction_op(mul_train.outputs["model"], mul_train.outputs["modfile"], tags.outputs["test_tags"])
        # xgb = gen_xgbpred_op(mul_train.outputs["model"], mul_train.outputs["modfile"],tags.outputs["train_tags"], tags.outputs["test_tags"])
    with dsl.Condition(arguments.output == 'iterative'):
        it_train = generate_ittrain_op(tags.outputs["train_tags"])
        it_train.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # prediction = gen_prediction_op(it_train.outputs["model"], it_train.outputs["modfile"], it_train.outputs["test_tags"])
        prediction = gen_prediction_op(it_train.outputs["model"], it_train.outputs["modfile"], tags.outputs["test_tags"])

kfp.Client(host=kfp_endpoint).create_run_from_pipeline_func(praxi_pipeline, arguments={})