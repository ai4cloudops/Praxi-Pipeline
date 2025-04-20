import os
import json
import time
import pickle
import gzip
import base64
import numpy as np
import xgboost as xgb
import scipy.sparse
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    hamming_loss
)

# --- Helper functions from your Flask server ---

def get_intersection(from_set, to_set):
    return [ele for ele in from_set if ele in to_set]

def tagsets_to_matrix(
    inference_flag=True,
    input_size=None, compact_factor=1,
    all_tags_l=None, tag_index_mapping=None,
    all_label_l=None, label_index_mapping=None,
    tags_by_instance_l=None, labels_by_instance_l=None,
    tagset_files=None, feature_importance=np.array([])
):
    op_durations = defaultdict(float)

    # --- timing for generating the matrix ---
    t_gen_mat_0 = time.time()
    t_get_feature_0 = time.time()

    # identify which tags to include
    used_idxs = np.where(feature_importance > 0)[0].tolist()
    used_tags_set = set(all_tags_l[idx] for idx in used_idxs if idx < len(all_tags_l))

    t_get_feature_t = time.time()
    op_durations["get_feature"] = t_get_feature_t - t_get_feature_0

    instance_row_list = []
    instance_row_idx_set = []
    # selector + mat_builder timing
    for i, instance_tags in enumerate(tags_by_instance_l):
        if input_size is None:
            input_size = len(all_tags_l) // compact_factor

        t_sel0 = time.time()
        used_instance_tags = get_intersection(used_tags_set, instance_tags)
        t_sel1 = time.time()
        op_durations["selector"] += (t_sel1 - t_sel0)

        if feature_importance.size and not used_instance_tags:
            continue

        # build each row
        row = np.zeros(input_size)
        t_bld0 = time.time()
        for tag in used_instance_tags:
            row[tag_index_mapping[tag] % input_size] = instance_tags[tag]
        # print("instance_row", row)
        t_bld1 = time.time()
        op_durations["mat_builder"] += (t_bld1 - t_bld0)

        instance_row_idx_set.append(i)
        instance_row_list.append(scipy.sparse.csr_matrix(row))

    # stack rows into a sparse feature matrix
    t_list0 = time.time()
    if instance_row_list:
        feature_matrix = scipy.sparse.vstack(instance_row_list)
    else:
        feature_matrix = scipy.sparse.csr_matrix([])
    t_list1 = time.time()
    op_durations["list_to_mat"] = t_list1 - t_list0

    t_gen_mat_t = time.time()
    op_durations["gen_mat"] = t_gen_mat_t - t_gen_mat_0

    # label matrix (only if not inference)
    label_matrix = np.array([])
    if not inference_flag:
        lbl_list = []
        for labels in labels_by_instance_l:
            row = np.zeros(len(all_label_l))
            for lbl in labels:
                if lbl in label_index_mapping:
                    row[label_index_mapping[lbl]] = 1
            lbl_list.append(row)
        if lbl_list:
            label_matrix = np.vstack(lbl_list)

    return (
        [tagset_files[i] for i in instance_row_idx_set],
        feature_matrix,
        label_matrix,
        instance_row_idx_set,
        len(instance_row_list),
        dict(op_durations)
    )

def one_hot_to_names(mapping_path, one_hot_matrix, mapping=None):
    if mapping is None:
        with open(mapping_path, 'rb') as fp:
            mapping = pickle.load(fp)
    idxs = np.nonzero(one_hot_matrix)
    out = defaultdict(list)
    for row_idx, col_idx in zip(*idxs):
        out[row_idx].append(mapping[col_idx])
    return out

def merge_preds(base, new, idx_map=None):
    if idx_map is None:
        for k, v in new.items():
            base[k].extend(v)
    else:
        for i, real_idx in enumerate(idx_map):
            base[real_idx].extend(new.get(i, []))
    return base

# --- Global config & cold-start model loading ---

dataset = "data_4"
n_models = 1000
shuffle_idx = 0
test_sample_batch_idx = 0
n_samples = 4
clf_njobs = 32
n_estimators = 100
depth = 1
input_size = None
dim_compact_factor = 1
tree_method = "exact"
max_bin = 1
with_filter = True
freq = 25

MODELS_PATH = os.environ.get("MODELS_PATH", "/opt/models")
models = []

def load_models():
    for i in range(n_models):
        mdir = (
            f"{MODELS_PATH}/cwd_ML_with_{dataset}_{n_models}_{i}_train_"
            f"{shuffle_idx}shuffleidx_{test_sample_batch_idx}testsamplebatchidx_"
            f"{n_samples}nsamples_{clf_njobs}njobs_{n_estimators}trees_"
            f"{depth}depth_{input_size}-{dim_compact_factor}rawinput_sampling1_"
            f"{tree_method}treemethod_{max_bin}maxbin_modize_par_{with_filter}"
            f"{freq}removesharedornoisestags_verpak"
        )
        model_json = f"{mdir}/model_init.json"

        if not os.path.isdir(mdir):
            raise FileNotFoundError(f"Model directory missing: {mdir}")
        if not os.path.isfile(model_json):
            raise FileNotFoundError(f"Model file missing: {model_json}")

        clf = xgb.XGBClassifier(
            max_depth=10, learning_rate=0.1, silent=False, objective='binary:logistic',
            booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1,
            max_delta_step=0, subsample=0.8, colsample_bytree=0.8,
            colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1
        )
        clf.load_model(model_json)

        with open(f"{mdir}/index_label_mapping", 'rb') as fp:
            mapping = pickle.load(fp)
        with open(f"{mdir}/tag_index_mapping", 'rb') as fp:
            tag_index_mapping = pickle.load(fp)
        with open(f"{mdir}/label_index_mapping", 'rb') as fp:
            label_index_mapping = pickle.load(fp)
        with open(f"{mdir}/index_tag_mapping", 'rb') as fp:
            all_tags_l = pickle.load(fp)
        with open(f"{mdir}/index_label_mapping", 'rb') as fp:
            all_label_l = pickle.load(fp)

        feature_importance = {
            idx: imp for idx, imp in enumerate(clf.feature_importances_)
        }

        models.append({
            "clf": clf,
            "mapping": mapping,
            "feature_importance": feature_importance,
            "all_tags_l": all_tags_l,
            "tag_index_mapping": tag_index_mapping,
            "all_label_l": all_label_l,
            "label_index_mapping": label_index_mapping
        })

load_models()

# --- Lambda entry point ---

def lambda_handler(event, context):
    # 1) Decode incoming body
    raw_body = event.get("body", "") or ""
    if event.get("isBase64Encoded", False):
        body_bytes = base64.b64decode(raw_body)
    else:
        body_bytes = raw_body.encode("utf-8")

    # 2) Decompress if gzipped
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    if headers.get("content-encoding") == "gzip":
        try:
            body_bytes = gzip.decompress(body_bytes)
        except OSError:
            # failed to decompress, assume it wasn't gzipped
            pass

    # 3) Parse JSON
    body = json.loads(body_bytes)

    # 4) Validate input
    samples = body.get("samples", [])
    if not samples:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No samples provided"})
        }

    # --- your existing prediction logic goes here ---
    instance_ids = [s["instance_id"] for s in samples]
    tags_by_instance = [s.get("tags", []) for s in samples]
    labels_by_instance = [s.get("true_labels", []) for s in samples]
    tagset_files = [str(i) for i in instance_ids]

    merged = defaultdict(list)
    encoder_metrics = {}

    for m_idx, m in enumerate(models):
        tfiles, feat_mat, lbl_mat, idx_map, cnt, ops = tagsets_to_matrix(
            inference_flag=False,
            input_size=input_size,
            compact_factor=dim_compact_factor,
            all_tags_l=m["all_tags_l"],
            tag_index_mapping=m["tag_index_mapping"],
            all_label_l=m["all_label_l"],
            label_index_mapping=m["label_index_mapping"],
            tags_by_instance_l=tags_by_instance,
            labels_by_instance_l=labels_by_instance,
            tagset_files=tagset_files,
            feature_importance=np.array(list(m["feature_importance"].values()))
        )

        if feat_mat.size:
            # np.set_printoptions(threshold=np.inf)  # disable truncation :contentReference[oaicite:5]{index=5}
            # print(feature_matrix.toarray())
            # dense_list = feat_mat.toarray().tolist()  # nested list :contentReference[oaicite:4]{index=4}
            # print("Literal form :", m["all_label_l"], repr(dense_list))

            t0 = time.time()
            preds = m["clf"].predict(feat_mat)
            # print("preds", preds)

            ops["predict_time"] = time.time() - t0
            ops["feature_matrix_size"]  = feat_mat.size
            ops["feature_matrix_xsize"] = feat_mat.shape[0]
            ops["feature_matrix_ysize"] = feat_mat.shape[1]
            names = one_hot_to_names(None, preds, mapping=m["mapping"])
            merged = merge_preds(merged, names, idx_map)
            # print("merged", merged)
        else:
            for i in range(len(samples)):
                merged.setdefault(i, [])

        encoder_metrics[f"model_{m_idx}"] = ops

    # compute overall metrics if any true_labels were provided
    metrics = {}
    if any(labels_by_instance):
        all_lbls = sorted(
            set(l for sub in labels_by_instance for l in sub) |
            set(l for sub in merged.values() for l in sub)
        )
        if all_lbls:
            mlb = MultiLabelBinarizer(classes=all_lbls)
            true_b = mlb.fit_transform(labels_by_instance)
            pred_ordered = [merged[i] for i in range(len(samples))]
            pred_b = mlb.transform(pred_ordered)

            metrics = {
                'accuracy':           accuracy_score(true_b, pred_b),
                'f1_score_weighted':  f1_score(true_b, pred_b, average='weighted'),
                'f1_score_macro':     f1_score(true_b, pred_b, average='macro'),
                'f1_score_micro':     f1_score(true_b, pred_b, average='micro'),
                'precision_weighted': precision_score(true_b, pred_b, average='weighted'),
                'precision_macro':    precision_score(true_b, pred_b, average='macro'),
                'precision_micro':    precision_score(true_b, pred_b, average='micro'),
                'recall_weighted':    recall_score(true_b, pred_b, average='weighted'),
                'recall_macro':       recall_score(true_b, pred_b, average='macro'),
                'recall_micro':       recall_score(true_b, pred_b, average='micro'),
                'hamming_loss':       hamming_loss(true_b, pred_b)
            }

    # 5) Build response payload
    response = {
        "predictions":     {iid: merged[idx] for idx, iid in enumerate(instance_ids)},
        "metrics":         metrics,
        "encoder_metrics": encoder_metrics
    }

    # 6) Compress & encode for API Gateway
    response_json = json.dumps(response)
    compressed = gzip.compress(response_json.encode("utf-8"))
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip"
        },
        "isBase64Encoded": True,
        "body": base64.b64encode(compressed).decode("utf-8")
    }

