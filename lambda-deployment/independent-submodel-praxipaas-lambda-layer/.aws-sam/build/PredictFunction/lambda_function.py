import os
import json
import time
import pickle
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

# --- Helper functions (from your Flask code) ---

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
    from collections import defaultdict
    import time
    import numpy as np
    import scipy.sparse

    op_durations = defaultdict(float)
    t0 = time.time()

    used_idxs = np.where(feature_importance > 0)[0].tolist()
    used_tags_set = set(all_tags_l[idx] for idx in used_idxs if idx < len(all_tags_l))

    instance_row_list = []
    instance_row_idx_set = []
    for i, instance_tags in enumerate(tags_by_instance_l):
        if input_size is None:
            input_size = len(all_tags_l)//compact_factor

        t_sel0 = time.time()
        used_instance_tags = get_intersection(used_tags_set, instance_tags)
        op_durations["selector"] += time.time() - t_sel0
        if feature_importance.size and not used_instance_tags:
            continue

        row = np.zeros(input_size)
        t_bld0 = time.time()
        for tag in used_instance_tags:
            row[tag_index_mapping[tag] % input_size] = instance_tags[tag]
        op_durations["mat_builder"] += time.time() - t_bld0

        instance_row_idx_set.append(i)
        instance_row_list.append(scipy.sparse.csr_matrix(row))

    if instance_row_list:
        feature_matrix = scipy.sparse.vstack(instance_row_list)
    else:
        feature_matrix = scipy.sparse.csr_matrix([])

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

    op_durations["total"] = time.time() - t0
    return (
        [tagset_files[i] for i in instance_row_idx_set],
        feature_matrix, label_matrix,
        instance_row_idx_set,
        len(instance_row_list),
        dict(op_durations)
    )

def one_hot_to_names(mapping_path, one_hot_matrix, mapping=None):
    if mapping is None:
        with open(mapping_path,'rb') as fp:
            mapping = pickle.load(fp)
    idxs = np.nonzero(one_hot_matrix)
    out = defaultdict(list)
    for r,c in zip(*idxs):
        out[r].append(mapping[c])
    return out

def merge_preds(base, new, idx_map=None):
    if idx_map is None:
        for k,v in new.items():
            base[k].extend(v)
    else:
        for i, real_idx in enumerate(idx_map):
            base[real_idx].extend(new.get(i, []))
    return base

# --- Cold-start model loading ---
models = []

def load_models():
    MODELS_PATH = os.environ.get("MODELS_PATH", "/opt/models")
    # Hard‑coded hyperparams (as before)
    n_models = 1000
    for i in range(n_models):
        mdir = f"{MODELS_PATH}/model_{i}"
        model_json = f"{mdir}/model_init.json"
        if not os.path.isfile(model_json):
            raise FileNotFoundError(model_json)

        clf = xgb.XGBClassifier()
        clf.load_model(model_json)

        # mappings
        with open(f"{mdir}/index_label_mapping",'rb') as fp:
            mapping = pickle.load(fp)
        with open(f"{mdir}/tag_index_mapping",'rb') as fp:
            tag_index_mapping = pickle.load(fp)
        with open(f"{mdir}/label_index_mapping",'rb') as fp:
            label_index_mapping = pickle.load(fp)
        with open(f"{mdir}/index_tag_mapping",'rb') as fp:
            all_tags_l = pickle.load(fp)
        with open(f"{mdir}/index_label_mapping",'rb') as fp:
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
    body = json.loads(event.get("body","{}"))
    samples = body.get("samples", [])
    if not samples:
        return {"statusCode":400, "body":json.dumps({"error":"No samples provided"})}

    instance_ids = [s["instance_id"] for s in samples]
    tags_by_instance = [ {t:1 for t in s.get("tags", [])} for s in samples ]
    labels_by_instance = [ s.get("true_labels",[]) for s in samples ]
    tagset_files = [str(i) for i in instance_ids]

    merged = defaultdict(list)
    encoder_metrics = {}

    for m_idx, m in enumerate(models):
        tfiles, feat_mat, lbl_mat, idx_map, cnt, ops = tagsets_to_matrix(
            inference_flag=False,
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
            t0 = time.time()
            preds = m["clf"].predict(feat_mat)
            ops["predict_time"] = time.time() - t0
            names = one_hot_to_names(None, preds, mapping=m["mapping"])
            merged = merge_preds(merged, names, idx_map)
        else:
            for i in range(len(samples)):
                merged.setdefault(i, [])

        encoder_metrics[f"model_{m_idx}"] = ops

    # compute overall metrics if true_labels exist
    metrics = {}
    if any(labels_by_instance):
        all_lbls = sorted(
            set(l for sub in labels_by_instance for l in sub) |
            set(l for sub in merged.values() for l in sub)
        )
        mlb = MultiLabelBinarizer(classes=all_lbls)
        true_b = mlb.fit_transform(labels_by_instance)
        pred_ordered = [merged[i] for i in range(len(samples))]
        pred_b = mlb.transform(pred_ordered)

        metrics = {
            "accuracy": accuracy_score(true_b,pred_b),
            "f1_macro": f1_score(true_b,pred_b,average="macro"),
            "precision_macro": precision_score(true_b,pred_b,average="macro"),
            "recall_macro": recall_score(true_b,pred_b,average="macro"),
            "hamming_loss": hamming_loss(true_b,pred_b)
        }

    response = {
        "predictions": {iid: merged[idx] for idx,iid in enumerate(instance_ids)},
        "metrics": metrics,
        "encoder_metrics": encoder_metrics
    }
    return {"statusCode":200, "body":json.dumps(response)}
