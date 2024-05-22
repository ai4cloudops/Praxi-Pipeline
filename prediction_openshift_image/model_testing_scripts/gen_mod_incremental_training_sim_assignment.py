import os, sys
from pathlib import Path
sys.path.insert(1, '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/function')
import json
import pickle
import time
import yaml
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import main
from hybrid_tags import Hybrid
args = main.get_inputs()


train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_SL/"
test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML/"
clustering_d = {}
clustering_d[0.999] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.999/name_groups.json"
clustering_d[0.99] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.99/name_groups.json"
clustering_d[0.98] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.98/name_groups.json"
clustering_d[0.95] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.95/name_groups.json"
clustering_d[0.9] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.9/name_groups.json"
clustering_d[0.85] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.85/name_groups.json"
clustering_d[0.8] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.8/name_groups.json"
clustering_d[0.75] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.75/name_groups.json"
clustering_d[0.7] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.7/name_groups.json"
clustering_d[0.65] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.65/name_groups.json"
clustering_d[0.6] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.6/name_groups.json"
clustering_d[0.55] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.55/name_groups.json"
clustering_d[0.5] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/sim_thr0.5/name_groups.json"


def get_intersection(from_set, to_set):
    ret = []
    for ele in from_set:
        if ele in to_set:
            ret.append(ele)
    return ret


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
    with ProcessPoolExecutor(max_workers=128) as executor:
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

# train_tags_batch = load_tag_files_concurrently(train_tags_path)
# train_tags = []
# train_tags.extend(train_tags_batch)
# train_tags.extend(train_tags_batch)
# train_tags.extend(train_tags_batch)
# train_tags.extend(train_tags_batch)
# train_tags.extend(train_tags_batch)
# train_tags.extend(train_tags_batch)
# train_tags.extend(train_tags_batch)
# train_tags.extend(train_tags_batch)
# train_tags.extend(train_tags_batch)
# train_tags.extend(train_tags_batch)
# train_tags.extend(train_tags_batch)
# train_tags.extend(train_tags_batch)

# test_tags = load_tag_files_concurrently(test_tags_path)

# vw_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/data/"
# with open(vw_tags_path+"train_tags.obj","wb") as filehandler:
#     pickle.dump(train_tags, filehandler)
# with open(vw_tags_path+"test_tags.obj","wb") as filehandler:
#     pickle.dump(test_tags, filehandler)

vw_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/data/"
with open(vw_tags_path+"train_tags.obj","rb") as filehandler:
    train_tags = pickle.load(filehandler)
with open(vw_tags_path+"test_tags.obj","rb") as filehandler:
    test_tags = pickle.load(filehandler)

for n_models in [1000]:
    for sim_thr in [0.95]:
        random_instance = random.Random(4)
        for shuffle_idx in range(10):
            cwd  = f"/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental/cwd_{n_models}_{sim_thr}_{shuffle_idx}/"
            Path(cwd).mkdir(parents=True, exist_ok=True)
            outdir  = f"{cwd}/results/"
            Path(outdir).mkdir(parents=True, exist_ok=True)
            args['outdir'] = outdir
            model_path = cwd+"pred_model.p"
            modfile_path = cwd+"model.vw"
            prediction_path = cwd+"test_result.txt"

            with open(clustering_d[sim_thr], 'r') as openfile:
                # Reading from json file
                name_groups = json.load(openfile)
            
            package_subset = name_groups["name_groups"]
            package_subset = random_instance.sample(package_subset, len(package_subset))
            # package_subset = [set([package.replace("==", "_v").replace(".","_") for package in train_subset]) for train_subset in package_subset]
            package_subset = [set([package for package in train_subset]) for train_subset in package_subset]

            regrouped = [set() for _ in range(min(n_models, len(package_subset)))]
            for i in range(0, len(package_subset)):
                regrouped[i%n_models].update(package_subset[i])


            for i, train_subset in enumerate(regrouped):
                local_train_tags = []
                for tags in train_tags:
                    # if tags["labels"] in train_subset:
                    if get_intersection(tags["labels"], train_subset):
                        local_train_tags.append(tags)
                if i == 0:
                    args['iterative'] = modfile_path
                    model = main.iterative_train(local_train_tags, args)
                    # model = main.multilabel_train(train_tags, args)
                    modfile = model.vw_modelfile
                    # os.popen('cp {0} {1}'.format(modfile, modfile_path))
                    with open(model_path, 'wb') as modfile:
                        pickle.dump(model, modfile)

                else:
                    args['previous'] = model_path
                    model = main.iterative_train(local_train_tags, args)
                    modfile = model.vw_modelfile
                    # os.popen('cp {0} {1}'.format(modfile, modfile_path))
                    with open(model_path, 'wb') as modfile:
                        pickle.dump(model, modfile)
                
                print(f"iteration {i} done ===============================")


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

