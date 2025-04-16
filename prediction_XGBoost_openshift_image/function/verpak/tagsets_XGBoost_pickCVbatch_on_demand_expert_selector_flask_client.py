#!/usr/bin/env python
import os
import sys
import time
import json
import pickle
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
import requests
from tqdm import tqdm
import yaml

# ------------------------
# Global Variables
# ------------------------
# Define cwd once as a global variable.
# Configuration parameters (adjust as needed)
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

# Build a working directory; this is passed as an argument to map_tagfilesl
cwd = (
    "/home/cc/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts_online/verification/"
    "cwd_ML_with_" + dataset + "_" + str(n_models) + "_train_" + str(shuffle_idx) +
    "shuffleidx_" + str(test_sample_batch_idx) + "testsamplebatchidx_" + str(n_samples) +
    "nsamples_" + str(n_jobs) + "njobs_" + str(clf_njobs) + "clfnjobs_" + str(n_estimators) +
    "trees_" + str(depth) + "depth_" + str(input_size) + "-" + str(dim_compact_factor) +
    "rawinput_sampling1_" + str(tree_method) + "treemethod_" + str(max_bin) +
    "maxbin_modize_par_" + str(with_filter) + f"{freq}removesharedornoisestags_verpak_on_demand_expert_flask_client/"
)
Path(cwd).mkdir(parents=True, exist_ok=True)

# ------------------------
# Helper Functions
# ------------------------
def build_logger(logger_name, logfilepath):
    import logging
    logger = logging.getLogger(logger_name)
    Path(logfilepath).mkdir(parents=True, exist_ok=True)
    f_handler = logging.FileHandler(filename=os.path.join(logfilepath, logger_name + '.log'))
    c_handler = logging.StreamHandler(stream=sys.stdout)
    f_handler.setLevel(logging.INFO)
    c_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    c_handler.setFormatter(f_format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)
    return logger

def map_tagfilesl(tags_path, tag_files, cwd, inference_flag, freq=100, tokens_filter_set=set()):
    all_tags_set, all_label_set = set(), set()
    tags_by_instance_l, labels_by_instance_l = [], []
    tagset_files = []
    data_instance_d_l = [
        read_tokens(tags_path, tag_file, cwd, inference_flag, freq=freq, tokens_filter_set=tokens_filter_set)
        for tag_file in tag_files
    ]
    for data_instance_d in data_instance_d_l:
        if len(data_instance_d) == 4:
            tagset_files.append(data_instance_d['tag_file'])
            all_tags_set.update(data_instance_d['local_all_tags_set'])
            tags_by_instance_l.append(data_instance_d['instance_feature_tags_d'])
            all_label_set.update(data_instance_d['labels'])
            labels_by_instance_l.append(data_instance_d['labels'])
    return {
        "tagset_files": tagset_files,
        "all_tags_set": all_tags_set,
        "tags_by_instance_l": tags_by_instance_l,
        "all_label_set": all_label_set,
        "labels_by_instance_l": labels_by_instance_l
    }

def read_tokens(tags_path, tag_file, cwd, inference_flag, freq=100, tokens_filter_set=set()):
    ret = {}
    ret["tag_file"] = tag_file
    try:
        with open(os.path.join(tags_path, tag_file), 'rb') as tf:
            local_all_tags_set = set()
            instance_feature_tags_d = defaultdict(int)
            tagset = yaml.load(tf, Loader=yaml.Loader)
            filtered_tags_l = list()
            for k, v in tagset['tags'].items():
                if k not in tokens_filter_set or tokens_filter_set.get(k, 0) < freq:
                    local_all_tags_set.add(k)
                    instance_feature_tags_d[k] += int(v)
                else:
                    filtered_tags_l.append(k)
            if local_all_tags_set == set():
                # Use the global cwd to build a logger
                logger = build_logger(tag_file, os.path.join(cwd, "logs"))
                logger.info('%s', f"{tag_file} has empty tags after filtering: {filtered_tags_l}")
                return ret
            ret["local_all_tags_set"] = local_all_tags_set
            ret["instance_feature_tags_d"] = instance_feature_tags_d
            if 'labels' in tagset:
                ret["labels"] = tagset['labels']
            else:
                ret["labels"] = [tagset['label']]
    except Exception as e:
        logger = build_logger(tag_file, os.path.join(cwd, "logs"))
        logger.info('%s', e)
    return ret

# -----------------------------------------------------------
# Data Loading Function
# -----------------------------------------------------------
def load_tagset_data():
    """
    Loads tagset data from a directory using the early version logic.
    Returns a list of sample dictionaries and operation durations.
    """
    op_durations = defaultdict(int)
    t0 = time.time()
    test_tags_path = "/home/cc/Praxi-Pipeline/data/data_4/tagset_ML_3_test_mini/"
    # test_tags_path = "/home/cc/Praxi-Pipeline/data/data_4/tagset_ML_3_test/"
    # test_tags_path = "/home/cc/Praxi-Pipeline/data/data_4/tagset_ML_3/"
    step = None  # will be defined after getting the list of files

    # Using the global cwd defined above.
    print("Using global cwd:", cwd)
    
    t_data_0 = time.time()
    tag_files = [f for f in os.listdir(test_tags_path) if f.endswith('tag')]
    tag_files_chunks = []
    step = len(tag_files) // 32 + 1
    for i in range(0, len(tag_files), step):
        tag_files_chunks.append(tag_files[i:i+step])

    pool = mp.Pool(processes=32)
    async_results = [
        pool.apply_async(map_tagfilesl, args=(test_tags_path, chunk, cwd, True, 25))
        for chunk in tqdm(tag_files_chunks, desc="Dispatching tasks")
    ]
    data_instance_list = []
    for async_res in tqdm(async_results, desc="Collecting results"):
        result = async_res.get()
        if result is not None:
            data_instance_list.append(result)
    pool.close()
    pool.join()

    all_tags_set, all_label_set = set(), set()
    tags_by_instance_l, labels_by_instance_l = [], []
    tagset_files = []
    for data_instance in data_instance_list:
        if len(data_instance) == 5:
            tagset_files.extend(data_instance['tagset_files'])
            all_tags_set.update(data_instance['all_tags_set'])
            tags_by_instance_l.extend(data_instance['tags_by_instance_l'])
            all_label_set.update(data_instance['all_label_set'])
            labels_by_instance_l.extend(data_instance['labels_by_instance_l'])

    t_data_t = time.time()
    op_durations["total_data_load_time"] = t_data_t - t_data_0
    total_time = time.time() - t0
    op_durations["total_time"] = total_time
    print("Data loading completed in {:.2f} seconds".format(total_time))
    print("DEBUG: Loaded {} instances".format(len(tags_by_instance_l)))

    samples = []
    for idx, instance_tags in enumerate(tags_by_instance_l):
        sample = {"instance_id": idx, "tags": instance_tags}
        if idx < len(labels_by_instance_l) and labels_by_instance_l[idx]:
            sample["true_labels"] = labels_by_instance_l[idx]
        samples.append(sample)

    return samples, op_durations

def send_request(samples):
    """
    Sends the collected samples to the Flask server endpoint for prediction.
    Returns the JSON response.
    """
    payload = {"samples": samples}
    # print("Payload to send:", json.dumps(payload, indent=2))
    # url = "http://localhost:5000/predict"  # Adjust if needed
    url = "http://3.134.116.205:5000/predict"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Error sending request:", e)
        sys.exit(1)

    return response.json()

if __name__ == "__main__":
    samples, durations = load_tagset_data()
    print("Number of samples loaded:", len(samples))
    
    result = send_request(samples)
    
    # Print the server response
    print("Server response:")
    print(json.dumps(result, indent=2))
    
    # Dump the server response to a JSON file using the global cwd.
    output_file = os.path.join(cwd, "server_response.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print("Server response dumped to:", output_file)
