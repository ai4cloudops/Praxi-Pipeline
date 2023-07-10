

kfp_endpoint='http://localhost:8080'

from typing import NamedTuple

import kfp
import kfp.dsl as dsl
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op

def generate_changesets(user_in: input(str), single: input(bool), multi: input(bool), cs_path: OutputPath(str), args_path: OutputPath(str)):
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
generate_changeset_op = kfp.components.create_component_from_func(generate_changesets, output_component_file='generate_changeset_component.yaml', base_image="lkorver/praxi-columbus-base:0.067")


#def generate_tagset(user_in: input(str), output_text_path: OutputPath(str), output_args_path: OutputPath(str)):
def generate_tagset(input_args_path: InputPath(str), changeset_path: InputPath(str), output_text_path: OutputPath(str), output_args_path: OutputPath(str)):
    '''generate tagset from the changeset'''
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
generate_tagset_op = kfp.components.create_component_from_func(generate_tagset, output_component_file='generate_tagset_component.yaml', base_image="lkorver/praxi-columbus-base:0.067")


def get_tags(user_in: input(str), train_tags_path: OutputPath(str), test_tags_path: OutputPath(str), output_args_path: OutputPath(str)):
    import tagset_gen
    import yaml
    from yaml import Loader
    import os
    import json
    import pickle
    #print(os.listdir('/pipelines/component/src/mix_test_tag'))
    train_tags = []
    print(os.listdir('/pipelines/component/src/demo_tagsets/mix_train_tag'))
    for tag_file in os.listdir('/pipelines/component/src/demo_tagsets/mix_train_tag'):
        if(tag_file[-3:] == 'tag'):
            with open('/pipelines/component/src/demo_tagsets/mix_train_tag/' + tag_file, 'rb') as tf:
                tag = yaml.load(tf, Loader = Loader)    
                train_tags.append(tag)
                # with open(train_tags_path, 'w') as tr_tags:
                #     tr_tags.write(json.dumps(tag) + '\n')
        
    with open(train_tags_path, 'wb') as writer:
        pickle.dump(train_tags, writer)
    print(len(train_tags))
    test_tags = []
    for tag_file in os.listdir('/pipelines/component/src/demo_tagsets/mix_test_tag'):
        if(tag_file[-3:] == 'tag'):
            with open('/pipelines/component/src/demo_tagsets/mix_test_tag/' + tag_file, 'rb') as tf:
                tag = yaml.load(tf, Loader = Loader)    
                test_tags.append(tag)
                # with open(test_tags_path, 'w') as ts_tags:
                #     ts_tags.write(json.dumps(tag) + '\n')
    with open(test_tags_path, 'wb') as writer:
        pickle.dump(test_tags, writer)
    print(len(test_tags))

    with open(output_args_path, 'wb') as argf:
        pickle.dump("multilabel", argf)
        
get_tagset_op = kfp.components.create_component_from_func(get_tags, output_component_file='get_tagset_component.yaml', base_image="lkorver/praxi-columbus-base:0.072")
    

def get_train_type(args_path: InputPath(str)) -> str:
    import tagset_gen
    import json
    import pickle
    with open(args_path, 'rb') as reader:
        user_in = pickle.load(reader)
    train_type = 'multilabel'
    return train_type
get_traintype_op = kfp.components.create_component_from_func(get_train_type, output_component_file='get_traintype_component.yaml', base_image="lkorver/praxi-columbus-base:0.065")
#

#def iterative_training(tagset_path: InputPath(str), modfile_path: OutputPath(str), model_path: OutputPath(str), test_tags_path: OutputPath(str)):
def iterative_training(tagset_path: InputPath(str), modfile_path: OutputPath(str), model_path: OutputPath(str)):
    ''''''
    import main
    import os
    import json
    import pickle
    import time
    args = main.get_inputs() 
    data_loaded = []

    with open(tagset_path, 'r') as stream:
        for line in stream:
            temp = json.loads(line)
            if (type(temp) != None):
                data_loaded.append(temp)
    
    model = main.iterative_train(data_loaded, args)
    modfile = model.vw_modelfile
    #print("modfile",modfile)
    os.popen('cp {0} {1}'.format(modfile, modfile_path))
    with open(model_path, 'wb') as modelfile:
        pickle.dump(model, modelfile)    
            
    # with open(test_tags_path, 'wb') as testfile:
    #     pickle.dump(testdat, testfile)
generate_ittrain_op = kfp.components.create_component_from_func(iterative_training, output_component_file='generate_ittrain_component.yaml', base_image="lkorver/praxi-vw-base:0.212") 


#def multilabel_training(tagset_path: InputPath(str), modfile_path: OutputPath(str), model_path: OutputPath(str)):
def multilabel_training(tagset_path: InputPath(str), created_tags_path: InputPath(str), modfile_path: OutputPath(str), model_path: OutputPath(str)):
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

    print(len(data_loaded))

    # with open(created_tags_path, 'r') as stream:
    #     for line in stream:
    #         temp = json.loads(line)
    #         if (type(temp) != None):
    #             data_loaded.append(temp)

    # print(len(data_loaded))
    
    model = main.multilabel_train(data_loaded, args)
    modfile = model.vw_modelfile
    os.popen('cp {0} {1}'.format(modfile, modfile_path))
    with open(model_path, 'wb') as modfile:
        pickle.dump(model, modfile)

    # with open(test_tags_path, 'wb') as testfile:
    #     pickle.dump(testdat, testfile)

generate_multilabel_op = kfp.components.create_component_from_func(multilabel_training, output_component_file='generate_multilabel_component.yaml', base_image="lkorver/praxi-vw-base:0.320")


#def gen_prediction(model_path: InputPath(str), modfile_path: InputPath(str), test_tags_path: InputPath(str), prediction_path: OutputPath(str)):
def gen_prediction(model_path: InputPath(str), modfile_path: InputPath(str), test_tags_path: InputPath(str), created_tags_path: InputPath(str), prediction_path: OutputPath(str)):
    '''generate prediction given model'''
    import main
    import os
    import json
    import pickle
    import time
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
    pred = main.test(model, data_loaded, args)
    print("output", pred)
    with open(prediction_path, 'wb') as writer:
        pickle.dump(pred, writer) 
    time.sleep(5000)
gen_prediction_op = kfp.components.create_component_from_func(gen_prediction, output_component_file='generate_pred_component.yaml', base_image="lkorver/praxi-vw-base:0.320") 


# Reading bigger data
@func_to_container_op
def print_text(text_path: InputPath()): # The "text" input is untyped so that any data can be printed
    '''Print text'''
    with open(text_path, 'rb') as reader:
        for line in reader:
            print(line, end = '')
    
def praxi_pipeline():
    # vop = dsl.VolumeOp(
    #     name="snapshot-claim-4",
    #     resource_name="snapshot-claim-4",
    #     size="1Gi",
    #     modes=dsl.VOLUME_MODE_RWM,
    #     volume_name="snapshot-volume",
    #     storage_class="manual",
    #     generate_unique_name=False,
    #     action='apply',
    #     set_owner_reference=True
    # )

    packages = "scikit-learn numpy"
    changeset = generate_changeset_op(packages, single=True, multi=True)
    # #changeset.execution_options.caching_strategy.max_cache_staleness = "P0D"
    tagset = generate_tagset_op(changeset.outputs["args"], changeset.outputs["cs"])
    # #tagset.execution_options.caching_strategy.max_cache_staleness = "P0D"

    change_test = generate_changeset_op("scoop scipy", single=False, multi=True)
    tag_test = generate_tagset_op(change_test.outputs["args"], change_test.outputs["cs"])

    tags = get_tagset_op(packages)
    
    #arguments = get_traintype_op(tagset.outputs["output_args"])
    arguments = get_traintype_op(tags.outputs["output_args"])
    with dsl.Condition(arguments.output == 'multilabel'):
        #mul_train = generate_multilabel_op(tagset.outputs["output_text"])
        mul_train = generate_multilabel_op(tags.outputs["train_tags"], tagset.outputs["output_text"])
        mul_train.execution_options.caching_strategy.max_cache_staleness = "P0D"
        #prediction = gen_prediction_op(mul_train.outputs["model"], mul_train.outputs["modfile"], tag_test.outputs["output_text"])
        prediction = gen_prediction_op(mul_train.outputs["model"], mul_train.outputs["modfile"], tags.outputs["test_tags"], tag_test.outputs["output_text"])
    # with dsl.Condition(arguments.output == 'iterative'):
        # it_train = generate_ittrain_op(tagset.outputs["output_text"])
        # it_train.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # #print_text(it_train.output)
        # #prediction = gen_prediction_op(it_train.outputs["model"], it_train.outputs["modfile"], it_train.outputs["test_tags"])
        # prediction = gen_prediction_op(it_train.outputs["model"], it_train.outputs["modfile"], tag_test.outputs["output_text"])
        # prediction.execution_options.caching_strategy.max_cache_staleness = "P0D"

kfp.Client(host=kfp_endpoint).create_run_from_pipeline_func(praxi_pipeline, arguments={})