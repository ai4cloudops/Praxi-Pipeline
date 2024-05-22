import os, sys
from pathlib import Path
sys.path.insert(1, '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/function')
import json
import pickle
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import main
from hybrid_tags import Hybrid
args = main.get_inputs()


# train_changes_path = "/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_train_tag/"
# test_changes_path = "/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/"
# train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/big_train/"
# test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/big_ML_biased_test/"
# test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/big_SL_biased_test/"
# test_tags_path = "/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/"
# train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_0_temp/big_train/"
# test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_0_temp/big_ML_biased_test/"
# train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML/"
# test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML/"
train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML_test/"
test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML_test/"
inc_train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML_test_one_ver/"
inc_test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML_test_one_ver/"
# inc_train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML_test_120/"
# inc_test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML_test_120/"
# train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_SL_test/"
# test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_SL_test/"
# train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_SL_test_0/tagsets_SL_test_one_ver/"
# test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML_test_one_ver/"
# train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_SL_test_0/tagsets_SL_test_one_ver/"
# test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_SL_test_0/tagsets_SL_test_one_ver/"
# train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_SL_test_0/tagsets_SL_test copy/"
# test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_SL_test_0/tagsets_SL_test copy/"
# train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_SL_one_ver"
# test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML_one_ver"
# train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/praxi/demos/ic2e_demo/demo_tagsets/sl_train_tag/"
# test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/praxi/demos/ic2e_demo/demo_tagsets/sl_test_tag/"
# train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/praxi/demos/ic2e_demo/demo_tagsets/ml_train_tag/"
# test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/praxi/demos/ic2e_demo/demo_tagsets/ml_test_tag/"

cwd  = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental/cwd/"
Path(cwd).mkdir(parents=True, exist_ok=True)
outdir  = f"{cwd}/results/"
Path(outdir).mkdir(parents=True, exist_ok=True)
args['outdir'] = outdir
model_path = cwd+"pred_model.p"
modfile_path = cwd+"model.vw"
prediction_path = cwd+"test_result.txt"


def load_tag_file(tag_file, tags_path):
    """Function to load a single tag file."""
    with open(os.path.join(tags_path, tag_file), 'rb') as tf:
        tag = yaml.load(tf, Loader=yaml.Loader)
    return tag

def load_tag_files_concurrently(tags_path):
    """Function to load tag files in parallel and return a list of tags."""
    # List to hold the contents of all tag files
    tags = []
    
    # Filter for files ending with '.tag'
    tag_files = [f for f in os.listdir(tags_path) if f.endswith('.tag')]
    
    # Use ThreadPoolExecutor to parallelize file loading
    with ProcessPoolExecutor(max_workers=20) as executor:
        # Create a future to file mapping
        future_to_tag_file = {executor.submit(load_tag_file, tag_file, tags_path): tag_file for tag_file in tag_files}
        
        # Process as each future completes
        for future in tqdm(as_completed(future_to_tag_file), total=len(tag_files), desc='Loading tag files'):
            try:
                tag = future.result()
                tags.append(tag)
            except Exception as e:
                print(f'Error loading file {future_to_tag_file[future]}: {e}')
    
    return tags

train_tags_batch = load_tag_files_concurrently(train_tags_path)
train_tags = []
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)

args['iterative'] = modfile_path
model = main.iterative_train(train_tags, args)
# model = main.multilabel_train(train_tags, args)
modfile = model.vw_modelfile
# os.popen('cp {0} {1}'.format(modfile, "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental/cwd/results"))
with open(model_path, 'wb') as modfile:
    pickle.dump(model, modfile)




test_tags = load_tag_files_concurrently(test_tags_path)
with open(model_path, 'rb') as reader:
    model = pickle.load(reader)
# model.vw_binary = 'docker run -v /home/cc/Praxi-study/vw-kubeflow-pipeline/Praxi-Pipeline/prediction_base_image:/workspace --rm vowpalwabbit/vw-rel-alpine:9.8.0'
# model.vw_modelfile = modfile_path
print("labs",model.all_labels)
pred = main.test(model, test_tags, args)
print("output", pred)
with open(prediction_path, 'wb') as writer:
    pickle.dump(pred, writer) 
# time.sleep(5000)

test_tags = load_tag_files_concurrently(inc_test_tags_path)
with open(model_path, 'rb') as reader:
    model = pickle.load(reader)
# model.vw_binary = 'docker run -v /home/cc/Praxi-study/vw-kubeflow-pipeline/Praxi-Pipeline/prediction_base_image:/workspace --rm vowpalwabbit/vw-rel-alpine:9.8.0'
model.vw_modelfile = modfile_path
print("labs",model.all_labels)
pred = main.test(model, test_tags, args)
print("output", pred)
with open(prediction_path, 'wb') as writer:
    pickle.dump(pred, writer) 
# time.sleep(5000)

# time.sleep(2)


train_tags_batch = load_tag_files_concurrently(inc_train_tags_path)
train_tags = []
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)
train_tags.extend(train_tags_batch)

args['previous'] = model_path
model = main.iterative_train(train_tags, args)
modfile = model.vw_modelfile
# os.popen('cp {0} {1}'.format(modfile, modfile_path))
with open(model_path, 'wb') as modfile:
    pickle.dump(model, modfile)




test_tags = load_tag_files_concurrently(test_tags_path)
with open(model_path, 'rb') as reader:
    model = pickle.load(reader)
# model.vw_binary = 'docker run -v /home/cc/Praxi-study/vw-kubeflow-pipeline/Praxi-Pipeline/prediction_base_image:/workspace --rm vowpalwabbit/vw-rel-alpine:9.8.0'
model.vw_modelfile = modfile_path
print("labs",model.all_labels)
pred = main.test(model, test_tags, args)
print("output", pred)
with open(prediction_path, 'wb') as writer:
    pickle.dump(pred, writer) 
# time.sleep(5000)

test_tags = load_tag_files_concurrently(inc_test_tags_path)
with open(model_path, 'rb') as reader:
    model = pickle.load(reader)
# model.vw_binary = 'docker run -v /home/cc/Praxi-study/vw-kubeflow-pipeline/Praxi-Pipeline/prediction_base_image:/workspace --rm vowpalwabbit/vw-rel-alpine:9.8.0'
model.vw_modelfile = modfile_path
print("labs",model.all_labels)
pred = main.test(model, test_tags, args)
print("output", pred)
with open(prediction_path, 'wb') as writer:
    pickle.dump(pred, writer) 
# time.sleep(5000)