#!/usr/bin/env python
import base64
import os
import sys
import time
import json
import pickle
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
import requests
import gzip
from tqdm import tqdm
import yaml
import numpy as np
import xgboost as xgb

# ------------------------
# Global Variables
# ------------------------
# Define cwd once as a global variable.
# Your Flask endpoint:
# FLASK_URL = "http://18.119.115.230:5000/predict"
FLASK_URL = "http://localhost:5000/predict"
# New: your API Gateway / Lambda POST endpoint
LAMBDA_URL = "https://ae2od52j3i.execute-api.us-east-2.amazonaws.com/Prod/predict"
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
cwd_clf = "/home/cc/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts_bak"


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
def load_all_used_tags(
    models_base_dir: str,
    dataset: str,
    n_models: int,
    shuffle_idx: int,
    test_sample_batch_idx: int,
    n_samples: int,
    clf_njobs: int,
    n_estimators: int,
    depth: int,
    input_size,
    dim_compact_factor: int,
    tree_method: str,
    max_bin: int,
    with_filter: bool,
    freq: int,
) -> set:
    used_tags = set()
    for i in range(n_models):
        # build the same path template you use in Lambda
        mdir = (
            f"{models_base_dir}/cwd_ML_with_{dataset}_{n_models}_{i}_train_"
            f"{shuffle_idx}shuffleidx_{test_sample_batch_idx}testsamplebatchidx_"
            f"{n_samples}nsamples_{clf_njobs}njobs_{n_estimators}trees_"
            f"{depth}depth_{input_size}-{dim_compact_factor}rawinput_sampling1_"
            f"{tree_method}treemethod_{max_bin}maxbin_modize_par_{with_filter}"
            f"{freq}removesharedornoisestags_verpak"
        )
        model_json = os.path.join(mdir, "model_init.json")
        idx2tag = os.path.join(mdir, "index_tag_mapping")

        if not os.path.isfile(model_json) or not os.path.isfile(idx2tag):
            continue

        # load model & feature importances
        clf = xgb.XGBClassifier(max_depth=10, learning_rate=0.1,silent=False, objective='binary:logistic', \
                booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1)
        clf.load_model(model_json)
        feats = clf.feature_importances_
        # load the index→tag list
        with open(idx2tag, "rb") as fp:
            all_tags_list = pickle.load(fp)

        # any feature with importance > 0
        for idx in np.where(feats > 0)[0]:
            if idx < len(all_tags_list):
                used_tags.add(all_tags_list[idx])
    return used_tags

def load_tagset_data(filtered_tags: set = None):
    """
    Loads tagset data from a directory using the early version logic.
    Returns a list of sample dictionaries and operation durations.
    """
    op_durations = defaultdict(int)
    t0 = time.time()
    test_tags_path = "/home/cc/Praxi-Pipeline/data/data_4/tagset_ML_3_test_mini-1/"
    # test_tags_path = "/home/cc/Praxi-Pipeline/data/data_4/tagset_ML_3_test_mini/"
    # test_tags_path = "/home/cc/Praxi-Pipeline/data/data_4/tagset_ML_3_test_mid/"
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
        if filtered_tags is not None:
            instance_tags = {
                tag: cnt
                for tag, cnt in instance_tags.items()
                if tag in filtered_tags
            }
        sample = {"instance_id": idx, "tags": instance_tags}
        if idx < len(labels_by_instance_l) and labels_by_instance_l[idx]:
            sample["true_labels"] = labels_by_instance_l[idx]
        samples.append(sample)

    return samples, op_durations

# def send_request(samples):
#     """
#     Sends samples to the Flask server.
#     """
#     payload = {"samples": samples}
#     headers = {"Content-Type": "application/json"}

#     try:
#         resp = requests.post(FLASK_URL, json=payload, headers=headers)
#         resp.raise_for_status()
#     except requests.exceptions.RequestException as e:
#         print("Error sending request to Flask:", e)
#         sys.exit(1)

#     return resp.json()

def send_request(samples):
    """
    Sends samples to the Flask server with gzip compression on request and response.
    """
    # 1) Build and compress payload
    payload = {"samples": samples}
    json_bytes = json.dumps(payload).encode("utf-8")
    compressed_request = gzip.compress(json_bytes)

    headers = {
        "Content-Type": "application/json",
        "Content-Encoding": "gzip",   # tell Flask we're sending gzipped body
        "Accept-Encoding": "gzip",    # ask Flask to gzip its response if possible
    }

    # 2) Send compressed bytes
    try:
        resp = requests.post(
            FLASK_URL,
            data=compressed_request,
            headers=headers,
            timeout=60
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Error sending request to Flask:", e)
        sys.exit(1)

    # 3) Let requests handle gzip decoding, just parse JSON
    try:
        return resp.json()
    except ValueError as e:
        print("Invalid JSON from Flask:", e)
        sys.exit(1)

    # # 3) Handle gzip‐compressed response
    # # If Flask responded with gzip, requests will expose raw bytes in resp.content
    # if resp.headers.get("Content-Encoding", "").lower() == "gzip":
    #     try:
    #         decompressed = gzip.decompress(resp.content)
    #         return json.loads(decompressed.decode("utf-8"))
    #     except Exception as e:
    #         print("Error decompressing Flask response:", e)
    #         sys.exit(1)
    # else:
    #     # Fallback to normal JSON parsing
    #     try:
    #         return resp.json()
    #     except ValueError as e:
    #         print("Invalid JSON from Flask:", e)
    #         sys.exit(1)


def send_to_lambda(samples):
    """
    Sends samples to the AWS Lambda endpoint via API Gateway,
    with gzip compression on request and response.
    """
    # 1) Build and compress payload
    payload = {"samples": samples}
    json_bytes = json.dumps(payload).encode("utf-8")
    compressed_request = gzip.compress(json_bytes)

    headers = {
        "Content-Type": "application/json",
        "Content-Encoding": "gzip",   # tell Lambda we're sending gzipped body
        # "x-api-key": "YOUR_API_KEY", # if required
    }

    # 2) Send compressed bytes
    try:
        resp = requests.post(
            LAMBDA_URL,
            data=compressed_request,
            headers=headers,
            timeout=60
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print("Error sending request to Lambda:", e)
        sys.exit(1)

    # 3) Parse top‐level JSON
    try:
        response_envelope = resp.json()
    except ValueError as e:
        print("Invalid JSON from Lambda:", e)
        sys.exit(1)

    # 4) Unwrap proxy integration envelope
    if isinstance(response_envelope, dict) and "body" in response_envelope:
        body = response_envelope["body"]

        # 5) If Lambda marked it base64‑encoded, decode & decompress
        if response_envelope.get("isBase64Encoded", False):
            try:
                compressed_response = base64.b64decode(body)
                decompressed = gzip.decompress(compressed_response)
                return json.loads(decompressed.decode("utf-8"))
            except Exception as e:
                print("Error decoding/decompressing Lambda response:", e)
                sys.exit(1)
        else:
            # plain JSON string inside `body`
            return json.loads(body)

    # 6) Fallback: return whatever we got
    return response_envelope


if __name__ == "__main__":
    used = load_all_used_tags(
        cwd_clf, dataset, n_models, shuffle_idx,
        test_sample_batch_idx, n_samples, clf_njobs,
        n_estimators, depth, input_size,
        dim_compact_factor, tree_method,
        max_bin, with_filter, freq
    )

    # pass it into your loaderf
    samples, durations = load_tagset_data(filtered_tags=used)
    print("Number of samples loaded:", len(samples))

    # 1) Send to your Flask app
    flask_result = send_request(samples)
    print("Flask response:")
    print(json.dumps(flask_result, indent=2))
    with open(os.path.join(cwd, "flask_response.json"), "w") as f:
        json.dump(flask_result, f, indent=2)
    print("Flask response dumped to:", os.path.join(cwd, "flask_response.json"))

    # # 2) Send to your Lambda via API Gateway
    # lambda_result = send_to_lambda(samples)
    # print("Lambda response:")
    # print(json.dumps(lambda_result, indent=2))
    # with open(os.path.join(cwd, "lambda_response.json"), "w") as f:
    #     json.dump(lambda_result, f, indent=2)
    # print("Lambda response dumped to:", os.path.join(cwd, "lambda_response.json"))
