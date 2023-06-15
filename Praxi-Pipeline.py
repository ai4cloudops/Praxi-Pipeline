

kfp_endpoint='http://localhost:8080'

from typing import NamedTuple

import kfp
import kfp.dsl as dsl
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op



# import data

# @func_to_container_op
# def generate_changeset(dir_to_scan: str, output_text_path: OutputPath(str)):
#     '''generate changeset by logging changes in the dir_to_scan'''
#     with open(output_text_path, 'w') as writer:
#         writer.write(dir_to_scan + '1\n')

# @func_to_container_op
# def generate_changeset(dir_to_scan: str, output_text_path: OutputPath(str)):
#     '''load differences from multiple snapshot layers'''
#     # changeset = "test_changeset"
#     # with open(output_text_path, 'w') as writer:
#     #     writer.write(changeset + '\n')
#     # import os, time
#     # dir_list = os.listdir("/fake-snapshot/")
#     # print("Files and directories in '", "/fake-snapshot/", "' :", dir_list)
#     # dir_list = os.listdir("/fake-snapshot/changesets")
#     # print("Files and directories in '", "/fake-snapshot/changesets", "' :", dir_list)
#     # # time.sleep(1000)

def generate_tagset(changeset_path: InputPath(str), output_text_path: OutputPath(str)):
    '''generate tagset from the changeset'''
    # import time
    # time.sleep(10000)
    import tagset_gen
    tag_dict_gen = tagset_gen.run()
    import json
    with open(output_text_path, 'w') as writer:
        for tag_dict in tag_dict_gen:
            writer.write(json.dumps(tag_dict) + '\n')
generate_tagset_op = kfp.components.create_component_from_func(generate_tagset, output_component_file='generate_tagset_component.yaml', base_image="lkorver/praxi-columbus-base:0.01")


def generate_prediction(tagset_path: InputPath(str), prediction_path: OutputPath(str)):
    '''generate prediction with tagset'''
    import main
    import os
    import json
    import pickle
    import time
    data_loaded = []
    with open(tagset_path, 'r') as stream:
        for line in stream:
            temp = json.loads(line)
            if (type(temp) != None):
                data_loaded.append(temp)
    pred_path = main.run(data_loaded)
    print("output", pred_path)
    with open(prediction_path, 'w') as writer:
        with open(pred_path, 'rb') as reader:
            pickle.load(reader)
                #writer.write(line + '\n')
    time.sleep(5000)

generate_prediction_op = kfp.components.create_component_from_func(generate_prediction, output_component_file='generate_prediction_component.yaml', base_image="lkorver/praxi-vw-base:0.046")


# Reading bigger data
@func_to_container_op
def print_text(text_path: InputPath()): # The "text" input is untyped so that any data can be printed
    '''Print text'''
    with open(text_path, 'r') as reader:
        for line in reader:
            print(line, end = '')

def praxi_pipeline():
    vop = dsl.VolumeOp(
        name="snapshot-claim-4",
        resource_name="snapshot-claim-4",
        size="1Gi",
        modes=dsl.VOLUME_MODE_RWM,
        volume_name="snapshot-volume",
        storage_class="manual",
        generate_unique_name=False,
        action='apply',
        set_owner_reference=True
    )
    dir_to_scan = ""
    # changeset = generate_changeset(dir_to_scan).add_pvolumes({"/fake-snapshot": vop.volume})
    # tagset = generate_tagset(changeset.outputs["output_text"]).add_pvolumes({"/fake-snapshot": vop.volume})
    tagset = generate_tagset_op("/").add_pvolumes({"/fake-snapshot": vop.volume})
    prediction = generate_prediction_op(tagset.outputs["output_text"]).add_pvolumes({"/fake-snapshot": vop.volume})
    prediction.execution_options.caching_strategy.max_cache_staleness = "P0D"
    print_text(prediction.output)

kfp.Client(host=kfp_endpoint).create_run_from_pipeline_func(praxi_pipeline, arguments={})