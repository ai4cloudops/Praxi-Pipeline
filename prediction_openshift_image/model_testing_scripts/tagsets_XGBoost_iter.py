import os, pickle
import yaml
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.datasets import make_multilabel_classification
import sklearn.metrics as metrics
from pathlib import Path

import xgboost as xgb


def tagsets_to_matrix(tags_path, cwd="/home/ubuntu/Praxi-Pipeline/prediction_base_image/model_testing_scripts/cwd/", train_flag=True, iter_flag=False):
    all_tags_set, all_label_set = set(), set()
    tags_by_instance_l, labels_by_instance_l = [], []
    tagset_files = []
    # print(os.listdir(tags_path))
    for tag_file in tqdm(os.listdir(tags_path)):
        if(tag_file[-3:] == 'tag'):
            with open(tags_path + tag_file, 'rb') as tf:
                # print(tag_file)
                tagset_files.append(tag_file)
                instance_feature_tags_d = defaultdict(int)
                tagset = yaml.load(tf, Loader=yaml.Loader)   

                # feature 
                for tag_vs_count in tagset['tags']:
                    k,v = tag_vs_count.split(":")
                    all_tags_set.add(k)
                    instance_feature_tags_d[k] += int(v)
                tags_by_instance_l.append(instance_feature_tags_d)

                # label
                if 'labels' in tagset:
                    all_label_set.update(tagset['labels'])
                    labels_by_instance_l.append(tagset['labels'])
                else:
                    all_label_set.add(tagset['label'])
                    labels_by_instance_l.append([tagset['label']])

    with open(cwd+'tagset_files.txt', 'w') as f:
        for line in tagset_files:
            f.write(f"{line}\n")

    # Feature Matrix Generation
    removed_tags_l = []
    ## Handling Mapping
    if train_flag and not iter_flag:  # generate initial mapping.
        all_tags_l = list(all_tags_set)
        tag_index_mapping = {}
        for idx, tag in enumerate(all_tags_l):
            tag_index_mapping[tag] = idx
        with open(cwd+'index_tag_mapping', 'wb') as fp:
            pickle.dump(all_tags_l, fp)
        with open(cwd+'tag_index_mapping', 'wb') as fp:
            pickle.dump(tag_index_mapping, fp)
    elif train_flag and iter_flag:  # adding mapping.
        with open(cwd+'index_tag_mapping', 'rb') as fp:
            loaded_all_tags_l = pickle.load(fp)
            loaded_all_tags_set = set(loaded_all_tags_l)
            new_tags_set = all_tags_set.difference(loaded_all_tags_set)
            all_tags_l = loaded_all_tags_l + list(new_tags_set)
            with open(cwd+'index_tag_mapping_iter', 'wb') as fp:
                pickle.dump(all_tags_l, fp)
        with open(cwd+'tag_index_mapping', 'rb') as fp:
            tag_index_mapping = pickle.load(fp)
            for idx, tag in enumerate(all_tags_l[len(loaded_all_tags_l):]):
                tag_index_mapping[tag] = idx+len(loaded_all_tags_l)
            with open(cwd+'tag_index_mapping_iter', 'wb') as fp:
                pickle.dump(tag_index_mapping, fp)
    elif not train_flag and iter_flag:  # load iter mapping.
        with open(cwd+'index_tag_mapping_iter', 'rb') as fp:
            all_tags_l = pickle.load(fp)
        with open(cwd+'tag_index_mapping_iter', 'rb') as fp:
            tag_index_mapping = pickle.load(fp)
    else:  # not train_flag and not iter_flag: load initial mapping.
        with open(cwd+'index_tag_mapping', 'rb') as fp:
            all_tags_l = pickle.load(fp)
        with open(cwd+'tag_index_mapping', 'rb') as fp:
            tag_index_mapping = pickle.load(fp)
    ## Handling Feature Matrix
    feature_matrix = np.zeros(len(all_tags_l))
    for instance_tags_d in tags_by_instance_l:
        instance_row = np.zeros(len(all_tags_l))
        for tag_name,tag_count in instance_tags_d.items():
            if tag_name in tag_index_mapping:  # remove new tags unseen in mapping.
                instance_row[tag_index_mapping[tag_name]] = tag_count
            else:
                removed_tags_l.append(tag_name)
        feature_matrix = np.vstack([feature_matrix, instance_row])
    feature_matrix = np.delete(feature_matrix, (0), axis=0)
    with open(cwd+'removed_tags_l', 'wb') as fp:
        pickle.dump(removed_tags_l, fp)
    with open(cwd+'removed_tags_l.txt', 'w') as f:
        for line in removed_tags_l:
            f.write(f"{line}\n")
    


    # Label Matrix Generation
    removed_label_l = []
    ## Handling Mapping
    if train_flag and not iter_flag:  # generate initial mapping.
        all_label_l = list(all_label_set)
        label_index_mapping = {}
        for idx, tag in enumerate(all_label_l):
            label_index_mapping[tag] = idx
        with open(cwd+'index_label_mapping', 'wb') as fp:
            pickle.dump(all_label_l, fp)
        with open(cwd+'label_index_mapping', 'wb') as fp:
            pickle.dump(label_index_mapping, fp)
        # with open(cwd+'removed_tags_l.txt', 'w') as f:
        #     for line in all_label_l:
        #         f.write(f"{line}\n")
    elif train_flag and iter_flag:  # adding mapping.
        with open(cwd+'index_label_mapping', 'rb') as fp:
            loaded_all_label_l = pickle.load(fp)
            loaded_all_label_set = set(loaded_all_label_l)
            new_label_set = all_label_set.difference(loaded_all_label_set)
            all_label_l = loaded_all_label_l + list(new_label_set)
            with open(cwd+'index_label_mapping_iter', 'wb') as fp:
                pickle.dump(all_label_l, fp)
        with open(cwd+'label_index_mapping', 'rb') as fp:
            label_index_mapping = pickle.load(fp)
            for idx, tag in enumerate(all_label_l[len(loaded_all_label_l):]):
                label_index_mapping[tag] = idx+len(loaded_all_label_l)
            with open(cwd+'label_index_mapping_iter', 'wb') as fp:
                pickle.dump(label_index_mapping, fp)
    elif not train_flag and iter_flag:  # load iter mapping.
        with open(cwd+'index_label_mapping_iter', 'rb') as fp:
            all_label_l = pickle.load(fp)
        with open(cwd+'label_index_mapping_iter', 'rb') as fp:
            label_index_mapping = pickle.load(fp)
        with open(cwd+'loaded_index_label_mapping_iter.txt', 'w') as f:
            for line in all_label_l:
                f.write(f"{line}\n")
    else:  # not train_flag and not iter_flag: load initial mapping.
        with open(cwd+'index_label_mapping', 'rb') as fp:
            all_label_l = pickle.load(fp)
        with open(cwd+'label_index_mapping', 'rb') as fp:
            label_index_mapping = pickle.load(fp)
        with open(cwd+'loaded_index_label_mapping.txt', 'w') as f:
            for line in all_label_l:
                f.write(f"{line}\n")
    ## Handling Label Matrix
    label_matrix = np.zeros(len(all_label_l))
    for labels in labels_by_instance_l:
        instance_row = np.zeros(len(all_label_l))
        for label in labels:
            if label in label_index_mapping:    # remove new labels
                instance_row[label_index_mapping[label]] = 1
            else:
                removed_label_l.append(label)
        label_matrix = np.vstack([label_matrix, instance_row])
    label_matrix = np.delete(label_matrix, (0), axis=0)
    with open(cwd+'removed_label_l', 'wb') as fp:
        pickle.dump(removed_label_l, fp)
    with open(cwd+'removed_label_l.txt', 'w') as f:
        for line in removed_label_l:
            f.write(f"{line}\n")
    
    return tagset_files, feature_matrix, label_matrix
    # return tagset_files, None, label_matrix


def print_metrics(cwd, outfile, y_true, y_pred, labels):
    # with open(cwd+'index_label_mapping', 'rb') as fp:
    #     labels = np.array(pickle.load(fp))
    report = metrics.classification_report(y_true, y_pred, target_names=labels)
    f1w = metrics.f1_score(y_true, y_pred, average='weighted')
    f1i = metrics.f1_score(y_true, y_pred, average='micro')
    f1a = metrics.f1_score(y_true, y_pred, average='macro')
    pw = metrics.precision_score(y_true, y_pred, average='weighted')
    pi = metrics.precision_score(y_true, y_pred, average='micro')
    pa = metrics.precision_score(y_true, y_pred, average='macro')
    rw = metrics.recall_score(y_true, y_pred, average='weighted')
    ri = metrics.recall_score(y_true, y_pred, average='micro')
    ra = metrics.recall_score(y_true, y_pred, average='macro')

    # os.makedirs(str(outdir), exist_ok=True)

    # if numfolds == 1:
    #     file_header = ("MULTILABEL EXPERIMENT REPORT\n" +
    #         time.strftime("Generated %c\n") +
    #         ('\nArgs: {}\n\n'.format(args) if args else '') +
    #         "EXPERIMENT WITH {} TEST CHANGESETS\n".format(len(y_true)))
    #     fstub = 'multi_exp'
    # else:
    #     file_header = ("MULTILABEL EXPERIMENT REPORT\n" +
    #         time.strftime("Generated %c\n") +
    #         ('\nArgs: {}\n'.format(args) if args else '') +
    #         "\n{} FOLD CROSS VALIDATION WITH {} CHANGESETS\n".format(numfolds, len(y_true)))
    #     fstub = 'multi_exp_cv'

    # if result_type == 'summary':
    #     file_header += ("F1 SCORE : {:.3f} weighted\n".format(f1w) +
    #         "PRECISION: {:.3f} weighted\n".format(pw) +
    #         "RECALL   : {:.3f} weighted\n".format(rw))
    #     fstub += '_summary'
    # else:
    #     file_header += ("F1 SCORE : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(f1w, f1i, f1a) +
    #         "PRECISION: {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(pw, pi, pa) +
    #         "RECALL   : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n\n".format(rw, ri, ra))
    #     file_header += (" {:-^55}\n".format("CLASSIFICATION REPORT") + report.replace('\n', "\n"))
    # # fname = get_free_filename(fstub, outdir, '.txt')

    file_header = ("F1 SCORE : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(f1w, f1i, f1a) +
            "PRECISION: {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n".format(pw, pi, pa) +
            "RECALL   : {:.3f} weighted, {:.3f} micro-avg'd, {:.3f} macro-avg'd\n\n".format(rw, ri, ra))
    file_header += (" {:-^55}\n".format("CLASSIFICATION REPORT") + report.replace('\n', "\n"))
    
    
    np.savetxt("{}".format(cwd+outfile),
            np.array([]), fmt='%d', header=file_header, delimiter=',',
            comments='')

if __name__ == "__main__":


#     X, y = make_multilabel_classification(
#     n_samples=32, n_classes=5, n_labels=3, random_state=0
# )


    train_tags_init_path = "/home/ubuntu/Praxi-Pipeline/data/demo_tagsets_mostly_multi_label_ter/mix_train_tag_init/"
    train_tags_iter_path = "/home/ubuntu/Praxi-Pipeline/data/demo_tagsets_mostly_multi_label_ter/mix_train_tag_iter/"
    # test_tags_path = "/home/ubuntu/Praxi-Pipeline/data/demo_tagsets_mostly_multi_label/mix_test_tag/"
    test_tags_path = "/home/ubuntu/Praxi-Pipeline/data/demo_tagsets_mostly_multi_label_ter/small_mix_test_tag/"
    cwd  ="/home/ubuntu/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/cwd/"
    Path(cwd).mkdir(parents=True, exist_ok=True)
    

    # Data
    train_tagset_files_init, train_feature_matrix_init, train_label_matrix_init = tagsets_to_matrix(train_tags_init_path, cwd=cwd, train_flag=True, iter_flag=False)
    train_tagset_files_iter, train_feature_matrix_iter, train_label_matrix_iter = tagsets_to_matrix(train_tags_iter_path, cwd=cwd, train_flag=True, iter_flag=True)
    test_tagset_files_init, test_feature_matrix_init, test_label_matrix_init = tagsets_to_matrix(test_tags_path, cwd=cwd, train_flag=False, iter_flag=False)
    test_tagset_files_iter, test_feature_matrix_iter, test_label_matrix_iter = tagsets_to_matrix(test_tags_path, cwd=cwd, train_flag=False, iter_flag=True)

    # Init Training & Testing
    BOW_XGB = xgb.XGBClassifier(max_depth=10, learning_rate=0.1,silent=False, objective='binary:logistic', \
                      booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                      subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1)
    BOW_XGB.fit(train_feature_matrix_init, train_label_matrix_init)
    pred_label_matrix_init = BOW_XGB.predict(test_feature_matrix_init)
    pred_label_prob_matrix_init = BOW_XGB.predict_proba(test_feature_matrix_init)

    # np.savetxt(cwd+'test_feature_matrix.out', test_feature_matrix, delimiter=',')

    np.savetxt(cwd+'pred_label_matrix_init.out', pred_label_matrix_init, delimiter=',')
    np.savetxt(cwd+'test_label_matrix_init.out', test_label_matrix_init, delimiter=',')
    np.savetxt(cwd+'pred_label_prob_matrix_init.out', pred_label_prob_matrix_init, delimiter=',')

    with open(cwd+'index_label_mapping', 'rb') as fp:
        labels = np.array(pickle.load(fp))
    print_metrics(cwd, 'metrics_init.out', test_label_matrix_init, pred_label_matrix_init, labels)

    BOW_XGB.save_model(cwd+'model_init.json')


    # Iter Training & Testing
    BOW_XGB = xgb.XGBClassifier(max_depth=10, learning_rate=0.1,silent=False, objective='binary:logistic', \
                      booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                      subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1)
    # BOW_XGB.load_model(cwd+'model_init.model')
    # BOW_XGB.fit(train_feature_matrix_iter, train_label_matrix_iter, xgb_model=BOW_XGB.get_booster())
    BOW_XGB.fit(train_feature_matrix_iter, train_label_matrix_iter, xgb_model=cwd+'model_init.model')
    pred_label_matrix_iter = BOW_XGB.predict(test_feature_matrix_iter)
    pred_label_prob_matrix_iter = BOW_XGB.predict_proba(test_feature_matrix_iter)

    # np.savetxt(cwd+'test_feature_matrix.out', test_feature_matrix, delimiter=',')

    np.savetxt(cwd+'pred_label_matrix_iter.out', pred_label_matrix_iter, delimiter=',')
    np.savetxt(cwd+'test_label_matrix_iter.out', test_label_matrix_iter, delimiter=',')
    np.savetxt(cwd+'pred_label_prob_matrix_iter.out', pred_label_prob_matrix_iter, delimiter=',')

    with open(cwd+'index_label_mapping_iter', 'rb') as fp:
        labels = np.array(pickle.load(fp))
    print_metrics(cwd, 'metrics_iter.out', test_label_matrix_iter, pred_label_matrix_iter, labels)

    
    # print(pred_label_matrix)


    # verify the init trees are still tehere.
    #   Preconfigure significantly large feature and label spaces
    #       Both VW and XGBoost might have this problem
    #       NN seems to be an easier way for fast incremental training, i.e., simply replace the output and input layer, until your model is not strong enough.
    # evaluation steps
    # compare VW and XGBoost
    # TF-IDF
    # NIW: 