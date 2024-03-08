import os, pickle, time, gc
import yaml, json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.datasets import make_multilabel_classification
import sklearn.metrics as metrics
from sklearn.preprocessing import Normalizer
import random
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp

import xgboost as xgb

import psutil, sys


def build_logger(logger_name, logfilepath):
    import logging
    # Create a custom logger
    logger = logging.getLogger(logger_name)

    # Create handlers
    # print(logdirpath)
    Path(logfilepath).mkdir(parents=True, exist_ok=True)
    f_handler = logging.FileHandler(filename=logfilepath+logger_name+'.log')
    c_handler = logging.StreamHandler(stream=sys.stdout)
    f_handler.setLevel(logging.INFO)
    c_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    c_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.setLevel(logging.INFO)
    
    return logger

def map_tagfilesl(tags_path, tag_files, cwd, inference_flag, tokens_filter_set=set()):
    all_tags_set, all_label_set = set(), set()
    tags_by_instance_l, labels_by_instance_l = [], []
    tagset_files = []
    data_instance_d_l = [read_tokens(tags_path, tag_file, cwd, inference_flag, tokens_filter_set=tokens_filter_set) for tag_file in tag_files]
    # data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
    for data_instance_d in data_instance_d_l:
        if len(data_instance_d) ==4:
                tagset_files.append(data_instance_d['tag_file'])
                all_tags_set.update(data_instance_d['local_all_tags_set'])
                tags_by_instance_l.append(data_instance_d['instance_feature_tags_d'])
                all_label_set.update(data_instance_d['labels'])
                labels_by_instance_l.append(data_instance_d['labels'])
    return {"tagset_files": tagset_files, "all_tags_set": all_tags_set, "tags_by_instance_l":tags_by_instance_l ,"all_label_set":all_label_set, "labels_by_instance_l":labels_by_instance_l}

def read_tokens(tags_path, tag_file, cwd, inference_flag, tokens_filter_set=set()):
    ret = {}
    ret["tag_file"] = tag_file
    try:
        # if(tag_file[-3:] == 'tag') and (tag_file[:-3].rsplit('-', 1)[0] in packages_select_set or packages_select_set == set()):
        with open(tags_path + tag_file, 'rb') as tf:
            # print(tag_file)
            # tagset_files.append(tag_file)
            local_all_tags_set = set()
            instance_feature_tags_d = defaultdict(int)
            tagset = yaml.load(tf, Loader=yaml.Loader)
            # tagset = json.load(tf)   

            # feature 
            filtered_tags_l = list()
            for tag_vs_count in tagset['tags']:
                k,v = tag_vs_count.split(":")
                if k not in tokens_filter_set:
                    local_all_tags_set.add(k)
                    instance_feature_tags_d[k] += int(v)
                else:
                    filtered_tags_l.append(k)
            if local_all_tags_set == set():
                logger = build_logger(tag_file, cwd+"logs/")
                logger.info('%s', tag_file+" has empty tags after filtering: "+str(filtered_tags_l))
                return ret
            ret["local_all_tags_set"] = local_all_tags_set
            ret["instance_feature_tags_d"] = instance_feature_tags_d
            # tags_by_instance_l.append(instance_feature_tags_d)

            # label
            if not inference_flag:
                if 'labels' in tagset:
                    ret["labels"] = tagset['labels']
                    # all_label_set.update(tagset['labels'])
                    # labels_by_instance_l.append(tagset['labels'])
                else:
                    ret["labels"] = [tagset['label']]
                    # all_label_set.add(tagset['label'])
                    # labels_by_instance_l.append([tagset['label']])
    except Exception as e: # work on python 3.x
        logger = build_logger(tag_file, cwd+"logs/")
        logger.info('%s', e)
    return ret

def tagsets_to_matrix(tags_path, tag_files_l = None, index_tag_mapping_path=None, tag_index_mapping_path=None, index_label_mapping_path=None, label_index_mapping_path=None, cwd="/home/cc/Praxi-study/Praxi-Pipeline/prediction_base_image/model_testing_scripts/cwd/", train_flag=False, inference_flag=True, iter_flag=False, packages_select_set=set(), tokens_filter_set=set(), input_size=None, compact_factor=1):
    if index_tag_mapping_path == None:
        index_tag_mapping_path=cwd+'index_tag_mapping'
        tag_index_mapping_path=cwd+'tag_index_mapping'
        index_label_mapping_path=cwd+'index_label_mapping'
        label_index_mapping_path=cwd+'label_index_mapping'

        index_tag_mapping_iter_path=cwd+"index_tag_mapping_iter"
        tag_index_mapping_iter_path=cwd+"tag_index_mapping_iter"
        index_label_mapping_iter_path=cwd+"index_label_mapping_iter"
        label_index_mapping_iter_path=cwd+"label_index_mapping_iter"
    
    all_tags_set, all_label_set = set(), set()
    tags_by_instance_l, labels_by_instance_l = [], []
    tagset_files = []

    # # debuging with lcoal tagset files
    # for tag_file in tqdm(os.listdir(tags_path)):
    #     data_instance_d = read_tokens(tags_path, tag_file, cwd, packages_select_set, inference_flag)
    #     if len(data_instance_d) ==4:
    #         tagset_files.append(data_instance_d['tag_file'])
    #         all_tags_set.update(data_instance_d['local_all_tags_set'])
    #         tags_by_instance_l.append(data_instance_d['instance_feature_tags_d'])
    #         all_label_set.update(data_instance_d['labels'])
    #         labels_by_instance_l.append(data_instance_d['labels'])


    if tag_files_l == None:
        tag_files_l = [tag_file for tag_file in os.listdir(tags_path) if (tag_file[-3:] == 'tag') and (tag_file[:-4].rsplit('-', 1)[0] in packages_select_set or packages_select_set == set())]
    # return 
    tag_files_l_of_l, step = [], len(tag_files_l)//mp.cpu_count()+1
    for i in range(0, len(tag_files_l), step):
        tag_files_l_of_l.append(tag_files_l[i:i+step])
    pool = mp.Pool(processes=mp.cpu_count())
    # pool = mp.Pool(processes=5)
    data_instance_d_l = [pool.apply_async(map_tagfilesl, args=(tags_path, tag_files_l, cwd, inference_flag), kwds={"tokens_filter_set": tokens_filter_set}) for tag_files_l in tqdm(tag_files_l_of_l)]
    data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
    pool.close()
    pool.join()
    for data_instance_d in data_instance_d_l:
        if len(data_instance_d) == 5:
                tagset_files.extend(data_instance_d['tagset_files'])
                all_tags_set.update(data_instance_d['all_tags_set'])
                tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
                all_label_set.update(data_instance_d['all_label_set'])
                labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
    

    # pool = mp.Pool(processes=mp.cpu_count())
    # # pool = mp.Pool(processes=1)
    # data_instance_d_l = [pool.apply_async(read_tokens, args=(tags_path, tag_file, cwd, packages_select_set, inference_flag)) for tag_file in tqdm(os.listdir(tags_path))]
    # data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
    # for data_instance_d in data_instance_d_l:
    #     if len(data_instance_d) ==4:
    #             tagset_files.append(data_instance_d['tag_file'])
    #             all_tags_set.update(data_instance_d['local_all_tags_set'])
    #             tags_by_instance_l.append(data_instance_d['instance_feature_tags_d'])
    #             all_label_set.update(data_instance_d['labels'])
    #             labels_by_instance_l.append(data_instance_d['labels'])

    # # kfp / RHODS pipeline with intermediate data as dumps 
    # with open(tags_path, 'rb') as reader:
    #     tagsets_l = pickle.load(reader)
    #     for tagset in tagsets_l:
    #         instance_feature_tags_d = defaultdict(int)
    #         # feature 
    #         for tag_vs_count in tagset['tags']:
    #             k,v = tag_vs_count.split(":")
    #             all_tags_set.add(k)
    #             instance_feature_tags_d[k] += int(v)
    #         tags_by_instance_l.append(instance_feature_tags_d)
    #         # label
    #         if not inference_flag:
    #             if 'labels' in tagset:
    #                 all_label_set.update(tagset['labels'])
    #                 labels_by_instance_l.append(tagset['labels'])
    #             else:
    #                 all_label_set.add(tagset['label'])
    #                 labels_by_instance_l.append([tagset['label']])
        
    # with open(cwd+'tagset_files.yaml', 'w') as f:
    #     yaml.dump(tagset_files, f)
    #     # for line in tagset_files:
    #     #     f.write(f"{line}\n")

    # Sorting instances
    if not inference_flag:
        zipped = list(zip(tagset_files, tags_by_instance_l, labels_by_instance_l))
        zipped.sort(key=lambda x: x[0])
        tagset_files, tags_by_instance_l, labels_by_instance_l = zip(*zipped)
    else:
        zipped = list(zip(tagset_files, tags_by_instance_l))
        zipped.sort(key=lambda x: x[0])
        tagset_files, tags_by_instance_l = zip(*zipped)



# #############
#     # Save tag:count in mapping format
#     with open(tags_path+"all_tags_set.obj","wb") as filehandler:
#          pickle.dump(all_tags_set, filehandler)
#     with open(tags_path+"all_label_set.obj","wb") as filehandler:
#          pickle.dump(all_label_set, filehandler)
#     with open(tags_path+"tags_by_instance_l.obj","wb") as filehandler:
#          pickle.dump(tags_by_instance_l, filehandler)
#     with open(tags_path+"labels_by_instance_l.obj","wb") as filehandler:
#          pickle.dump(labels_by_instance_l, filehandler)
#     with open(tags_path+"tagset_files.obj","wb") as filehandler:
#          pickle.dump(tagset_files, filehandler)

    # # Load tag:count in mapping format 
    # with open(tags_path+"all_tags_set.obj","rb") as filehandler:
    #     all_tags_set = pickle.load(filehandler)
    # with open(tags_path+"all_label_set.obj","rb") as filehandler:
    #     all_label_set = pickle.load(filehandler)
    # with open(tags_path+"tags_by_instance_l.obj","rb") as filehandler:
    #     tags_by_instance_l = pickle.load(filehandler)
    # with open(tags_path+"labels_by_instance_l.obj","rb") as filehandler:
    #     labels_by_instance_l = pickle.load(filehandler)
    # with open(tags_path+"tagset_files.obj","rb") as filehandler:
    #     tagset_files = pickle.load(filehandler)
# #############

    # Feature Matrix Generation
    removed_tags_l = []
    ## Generate Mapping
    if train_flag and not iter_flag:  # generate initial mapping.
        all_tags_l = list(all_tags_set)
        tag_index_mapping = {}
        for idx, tag in enumerate(all_tags_l):
            tag_index_mapping[tag] = idx
        with open(index_tag_mapping_path, 'wb') as fp:
            pickle.dump(all_tags_l, fp)
        with open(tag_index_mapping_path, 'wb') as fp:
            pickle.dump(tag_index_mapping, fp)
    elif train_flag and iter_flag:  # adding mapping.
        with open(index_tag_mapping_path, 'rb') as fp:
            loaded_all_tags_l = pickle.load(fp)
            loaded_all_tags_set = set(loaded_all_tags_l)
            new_tags_set = all_tags_set.difference(loaded_all_tags_set)
            all_tags_l = loaded_all_tags_l + list(new_tags_set)
            with open(index_tag_mapping_iter_path, 'wb') as fp:
                pickle.dump(all_tags_l, fp)
        with open(tag_index_mapping_path, 'rb') as fp:
            tag_index_mapping = pickle.load(fp)
            for idx, tag in enumerate(all_tags_l[len(loaded_all_tags_l):]):
                tag_index_mapping[tag] = idx+len(loaded_all_tags_l)
            with open(tag_index_mapping_iter_path, 'wb') as fp:
                pickle.dump(tag_index_mapping, fp)
    elif not train_flag and iter_flag:  # load iter mapping.
        with open(index_tag_mapping_iter_path, 'rb') as fp:
            all_tags_l = pickle.load(fp)
        with open(tag_index_mapping_iter_path, 'rb') as fp:
            tag_index_mapping = pickle.load(fp)
    else:  # not train_flag and not iter_flag: load initial mapping.
        with open(index_tag_mapping_path, 'rb') as fp:
            all_tags_l = pickle.load(fp)
        with open(tag_index_mapping_path, 'rb') as fp:
            tag_index_mapping = pickle.load(fp)
    ## Generate Feature Matrix
    instance_row_list = []
    for instance_tags_d in tags_by_instance_l:
        if input_size == None:
            input_size = len(all_tags_l)//compact_factor
        instance_row = np.zeros(input_size)
        for tag_name,tag_count in instance_tags_d.items():
            if tag_name in tag_index_mapping:  # remove new tags unseen in mapping.
                instance_row[tag_index_mapping[tag_name]%input_size] = tag_count
            else:
                removed_tags_l.append(tag_name)
        # if input_size == None:
        #     instance_row = np.zeros(len(all_tags_l))
        #     for tag_name,tag_count in instance_tags_d.items():
        #         if tag_name in tag_index_mapping:  # remove new tags unseen in mapping.
        #             instance_row[tag_index_mapping[tag_name]] = tag_count
        #         else:
        #             removed_tags_l.append(tag_name)
        # else:
        #     instance_row = np.zeros(input_size)
        #     for tag_name,tag_count in instance_tags_d.items():
        #         if tag_name in tag_index_mapping:  # remove new tags unseen in mapping.
        #             instance_row[tag_index_mapping[tag_name]%input_size] = tag_count
        #         else:
        #             removed_tags_l.append(tag_name)
        # else:
        #     # instance_row = np.zeros(input_size)
        #     instance_row = np.random.randint(1000000000, size=input_size)
        #     # instance_row = np.array(range(input_size-1,-1,-1))
        instance_row_list.append(instance_row)
    # instance_row_list.extend(instance_row_list)
    feature_matrix = np.vstack(instance_row_list)
    del instance_row_list
    # with open(cwd+'removed_tags_l', 'wb') as fp:
    #     pickle.dump(removed_tags_l, fp)
    # with open(cwd+'removed_tags_l.txt', 'w') as f:
    #     for line in removed_tags_l:
    #         f.write(f"{line}\n")
    


    # Label Matrix Generation
    label_matrix = np.array([])
    if not inference_flag:
        removed_label_l = []
        ## Handling Mapping

        if train_flag and not iter_flag:  # generate initial mapping.
            all_label_l = list(all_label_set)
            label_index_mapping = {}
            for idx, tag in enumerate(all_label_l):
                label_index_mapping[tag] = idx
            with open(index_label_mapping_path, 'wb') as fp:
                pickle.dump(all_label_l, fp)
            with open(label_index_mapping_path, 'wb') as fp:
                pickle.dump(label_index_mapping, fp)
            # with open(cwd+'removed_tags_l.txt', 'w') as f:
            #     for line in all_label_l:
            #         f.write(f"{line}\n")
        elif train_flag and iter_flag:  # adding mapping.
            with open(index_label_mapping_path, 'rb') as fp:
                loaded_all_label_l = pickle.load(fp)
                loaded_all_label_set = set(loaded_all_label_l)
                new_label_set = all_label_set.difference(loaded_all_label_set)
                all_label_l = loaded_all_label_l + list(new_label_set)
                with open(index_label_mapping_iter_path, 'wb') as fp:
                    pickle.dump(all_label_l, fp)
            with open(label_index_mapping_path, 'rb') as fp:
                label_index_mapping = pickle.load(fp)
                for idx, tag in enumerate(all_label_l[len(loaded_all_label_l):]):
                    label_index_mapping[tag] = idx+len(loaded_all_label_l)
                with open(label_index_mapping_iter_path, 'wb') as fp:
                    pickle.dump(label_index_mapping, fp)
        elif not train_flag and iter_flag:  # load iter mapping.
            with open(index_label_mapping_iter_path, 'rb') as fp:
                all_label_l = pickle.load(fp)
            with open(label_index_mapping_iter_path, 'rb') as fp:
                label_index_mapping = pickle.load(fp)
            with open(cwd+'loaded_index_label_mapping_iter.txt', 'w') as f:
                for line in all_label_l:
                    f.write(f"{line}\n")
        else:  # not train_flag and not iter_flag: load initial mapping.
            with open(index_label_mapping_path, 'rb') as fp:
                all_label_l = pickle.load(fp)
            with open(label_index_mapping_path, 'rb') as fp:
                label_index_mapping = pickle.load(fp)
            with open(cwd+'loaded_index_label_mapping.txt', 'w') as f:
                for line in all_label_l:
                    f.write(f"{line}\n")
        ## Handling Label Matrix
        instance_row_list = []
        # label_matrix = np.zeros(len(all_label_l))
        for labels in labels_by_instance_l:
            instance_row = np.zeros(len(all_label_l))
            for label in labels:
                if label in label_index_mapping:    # remove new labels
                    instance_row[label_index_mapping[label]] = 1
                else:
                    removed_label_l.append(label)
            instance_row_list.append(instance_row)
            # label_matrix = np.vstack([label_matrix, instance_row])
        # label_matrix = np.delete(label_matrix, (0), axis=0)
        # instance_row_list.extend(instance_row_list)
        label_matrix = np.vstack(instance_row_list)
        with open(cwd+'removed_label_l', 'wb') as fp:
            pickle.dump(removed_label_l, fp)
        with open(cwd+'removed_label_l.txt', 'w') as f:
            for line in removed_label_l:
                f.write(f"{line}\n")
    
    return tagset_files, feature_matrix, label_matrix
    # return tagset_files, None, label_matrix

def one_hot_to_names(mapping_path, one_hot_matrix):
    with open(mapping_path, 'rb') as fp:
        mapping = pickle.load(fp)
    idxs_yx = np.nonzero(one_hot_matrix)
    labels = defaultdict(list)
    for entry_idx, (row_idx, col_idx) in enumerate(zip(idxs_yx[0],idxs_yx[1])):
        labels[int(row_idx)].append(mapping[col_idx])
    return labels

def merge_preds(labels_1, labels_2):
    labels = defaultdict(list)
    for idx in labels_1.keys():
        labels[idx].extend(labels_1[idx])
    for idx in labels_2.keys():
        labels[idx].extend(labels_2[idx])
    return labels

def print_metrics(cwd, outfile, y_true, y_pred, labels, op_durations=None):
    # with open(cwd+'index_label_mapping', 'rb') as fp:
    #     labels = np.array(pickle.load(fp))
    print(len(y_true), len(y_true[0]), len(y_pred), len(labels))
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

    if op_durations != None:
        file_header += ("\n {:-^55}\n".format("DURATION REPORT") + "\n".join(["{}:{:.3f}".format(k, v) for k, v in op_durations.items()]))
    
    
    np.savetxt("{}".format(cwd+outfile),
            np.array([]), fmt='%d', header=file_header, delimiter=',',
            comments='')

def run_init_train(train_tags_init_path, test_tags_path, cwd, train_tags_init_l=None, test_tags_l=None, n_jobs=64, n_estimators=100, train_packages_select_set=set(), highlight_label_set=set(), tokens_filter_set=set(), test_packages_select_set=set(), test_batch_count=1, input_size=None, compact_factor=1, depth=1, tree_method="auto", max_bin=6):
    # train_tags_init_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_0&2/big_SL_biased_test/"
    # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_0&2/big_ML_biased_test/"
    # # cwd  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_init/"
    # cwd  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0&2_SL/"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    op_durations = {}
    
    # process = psutil.Process()
    # print(process.memory_info())

    BOW_XGB_init = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=depth, learning_rate=0.1,silent=False, objective='binary:logistic', \
                        booster='gbtree', n_jobs=n_jobs, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                        subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, tree_method=tree_method, max_bin=max_bin)
                        #   subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1, tree_method=tree_method)

    
    # Train Data
    t0 = time.time()
    train_tagset_files_init, train_feature_matrix_init, train_label_matrix_init = tagsets_to_matrix(train_tags_init_path, tag_files_l=train_tags_init_l, cwd=cwd, train_flag=True, inference_flag=False, packages_select_set=train_packages_select_set, tokens_filter_set=tokens_filter_set, input_size=input_size, compact_factor=compact_factor)
    # print(process.memory_info())
    t1 = time.time()
    print(t1-t0)
    op_durations["tagsets_to_matrix-trainset"] = t1-t0
    op_durations["tagsets_to_matrix-trainset_xsize"] = train_feature_matrix_init.shape[0]
    op_durations["tagsets_to_matrix-trainset_ysize"] = train_feature_matrix_init.shape[1]

    # ######## save train_feature_matrix_init
    # with open(cwd+"train_feature_matrix_init.mat","wb") as filehandler:
    #     np.save(filehandler, train_feature_matrix_init)

    # ######## Plot train feature usage as a B/W plot
    # fig, ax = plt.subplots(1, 1, figsize=(600, 10))
    # ax.imshow(train_feature_matrix_init > 0, cmap='hot', interpolation="nearest")
    # plt.savefig(cwd+'train_feature_matrix_init.pdf', format='pdf', dpi=50, bbox_inches='tight')
    # plt.close(fig)
    # gc.collect()

    ######## Plot average train feature usage per label as a B/W plot
    # train_feature_init_used_count = (train_feature_matrix_init > 0).sum(axis=0)
    # train_label_init_used_count = (train_label_matrix_init > 0).sum(axis=1)
    # label_count_d = defaultdict(int)
    # for line in train_tagset_files_init:
    #     label_count_d["-".join(line.split("-")[:-1])] += 1
    train_feature_init_used_count_list = []
    idxs_yx = np.nonzero(train_label_matrix_init)
    label_row_idx = np.array([])
    col_idx_prev = -1
    row_idx_prev = -1
    for entry_idx, (row_idx, col_idx) in enumerate(zip(idxs_yx[0],idxs_yx[1])):
        if col_idx_prev != col_idx:
            if col_idx_prev != -1:
                train_feature_init_used_count_list.append(train_feature_matrix_init[list(range(row_idx_prev, row_idx)), :].mean(axis=0))
            col_idx_prev = col_idx
            row_idx_prev = row_idx
            label_row_idx = np.append(label_row_idx, [row_idx])
        if entry_idx == len(idxs_yx[0])-1 and col_idx_prev != -1:
            train_feature_init_used_count_list.append(train_feature_matrix_init[list(range(row_idx_prev, row_idx+1)), :].mean(axis=0))
    train_feature_init_used_count = np.vstack(train_feature_init_used_count_list)
    # fig, ax = plt.subplots(1, 1, figsize=(600, 10))
    # ax.imshow(train_feature_init_used_count > 0, cmap='hot', interpolation="nearest")
    # plt.savefig(cwd+'train_feature_init_used_count.pdf', format='pdf', dpi=50, bbox_inches='tight')
    # plt.close(fig)
    # gc.collect()
    # fig, ax = plt.subplots(1, 1, figsize=(600, 10))
    # ax.bar(list(range(train_feature_matrix_init.shape[1])), (train_feature_init_used_count > 0).sum(axis=0))
    # plt.savefig(cwd+'train_feature_init_used_count_bar.pdf', format='pdf', dpi=50, bbox_inches='tight')
    # plt.close(fig)
    # gc.collect()
    unique, counts = np.unique((train_feature_init_used_count > 0).sum(axis=0), return_counts=True)
    feature_occur, feature_occurence_count = [], []
    for idx in range(min(unique), max(unique)+1):
        feature_occur.append(idx)
        if idx in unique:
            feature_occurence_count.extend(list(counts[np.where(unique == idx)]))
        else:
            feature_occurence_count.append(0)



    if len(highlight_label_set) != 0:
        with open(cwd+'label_index_mapping', 'rb') as fp:
            label_index_mapping = pickle.load(fp)
        highlight_label_train_feature_init_used_count_list = []
        for label in highlight_label_set:
            highlight_label_train_feature_init_used_count_list.append(train_feature_matrix_init[np.where(train_label_matrix_init[:,label_index_mapping[label]] == 1)].mean(axis=0))
        highlight_label_train_feature_init_used_count = np.vstack(highlight_label_train_feature_init_used_count_list)
        highlight_label_feature_occur, highlight_label_feature_occurence_count = np.unique((highlight_label_train_feature_init_used_count > 0).sum(axis=0), return_counts=True)



    

    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=18)
    # feature_occurence_count_normalized = [round(feature_occurence_count_entry/sum(feature_occurence_count)*100, 2) for feature_occurence_count_entry in feature_occurence_count]
    # p = ax.bar(feature_occur,feature_occurence_count_normalized)
    # ax.bar_label(p, fontsize=18)
    # # ax.set_xticklabels([str(idx) for idx in x])
    # # ax.set_xticks(x)
    # ax.set_xlim([-0.5,5.5])
    # ax.set_title("% of Tokens Occurring in Multiple Packages", fontsize=20)
    # ax.set_ylabel("% of Tokens", fontsize=20)
    # ax.set_xlabel("Number of Packages", fontsize=20)
    # plt.savefig(cwd+'train_feature_init_used_count_freq_bar.pdf', format='pdf', dpi=50, bbox_inches='tight')
    # plt.close(fig)
    # gc.collect()

    # ######## Plot train label occurance as a B/W plot
    # fig, ax = plt.subplots(1, 1, figsize=(600, 10))
    # ax.imshow(train_label_matrix_init > 0, cmap='hot', interpolation="nearest")
    # plt.savefig(cwd+'train_label_matrix_init.pdf', format='pdf', dpi=50, bbox_inches='tight')
    # plt.close(fig)
    # gc.collect()

    # with open(cwd+'train_tagset_files_init.txt', 'w') as f:
    #     for line in train_tagset_files_init:
    #         f.write(f"{line}\n")

    # Training
    t0 = time.time()
    BOW_XGB_init.fit(train_feature_matrix_init, train_label_matrix_init)
    t1 = time.time()
    op_durations["BOW_XGB_init.fit"] = t1-t0
    BOW_XGB_init.save_model(cwd+'model_init.json')

    # Data Distribution Summary
    if len(highlight_label_set) != 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        zipped = list(zip(highlight_label_feature_occur, highlight_label_feature_occurence_count))
        zipped.sort(key=lambda x: x[1],reverse=True)
        highlight_label_feature_occur, highlight_label_feature_occurence_count = zip(*zipped)
        highlight_label_feature_occurence_count_normalized = [round(highlight_label_feature_occurence_count_entry/sum(highlight_label_feature_occurence_count)*100, 2) for highlight_label_feature_occurence_count_entry in highlight_label_feature_occurence_count]
        p = ax.bar([idx for idx in range(len(highlight_label_feature_occurence_count_normalized))],highlight_label_feature_occurence_count_normalized)
        ax.bar_label(p, fontsize=18)
        ax.set_xticklabels([str(occur) for occur in highlight_label_feature_occur])
        ax.set_xticks([idx for idx in range(len(highlight_label_feature_occurence_count_normalized))])
        ax.set_xlim([-0.5,5.5])
        ax.set_title("% of Tokens Occurring in Multiple Packages", fontsize=20)
        ax.set_ylabel("% of Tokens", fontsize=20)
        ax.set_xlabel("Number of Packages", fontsize=20)
        plt.savefig(cwd+'highlight_label_train_feature_init_used_count_freq_bar.pdf', format='pdf', dpi=50, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    zipped = list(zip(feature_occur, feature_occurence_count))
    zipped.sort(key=lambda x: x[1],reverse=True)
    feature_occur, feature_occurence_count = zip(*zipped)
    feature_occurence_count_normalized = [round(feature_occurence_count_entry/sum(feature_occurence_count)*100, 2) for feature_occurence_count_entry in feature_occurence_count]
    p = ax.bar([idx for idx in range(len(feature_occurence_count_normalized))],feature_occurence_count_normalized)
    ax.bar_label(p, fontsize=18)
    ax.set_xticklabels([str(occur) for occur in feature_occur])
    ax.set_xticks([idx for idx in range(len(feature_occurence_count_normalized))])
    ax.set_xlim([-0.5,5.5])
    ax.set_title("% of Tokens Occurring in Multiple Packages", fontsize=20)
    ax.set_ylabel("% of Tokens", fontsize=20)
    ax.set_xlabel("Number of Packages", fontsize=20)
    plt.savefig(cwd+'train_feature_init_used_count_freq_bar.pdf', format='pdf', dpi=50, bbox_inches='tight')
    plt.close(fig)
    gc.collect()





    # Testing Epochs
    # with open(cwd+'index_label_mapping', 'rb') as fp:
    #     labels = np.array(pickle.load(fp))
    label_matrix_list, pred_label_matrix_list = [], []
    if test_tags_l == None:
        test_tags_l = [tag_file for tag_file in os.listdir(test_tags_path) if tag_file[-3:] == 'tag']
    step = len(test_tags_l)//test_batch_count+1
    for batch_first_idx in range(0, len(test_tags_l), step):
        # Test Data
        t0 = time.time()
        test_tagset_files_init, test_feature_matrix_init, test_label_matrix_init = tagsets_to_matrix(test_tags_path, tag_files_l=test_tags_l[batch_first_idx:batch_first_idx+step], cwd=cwd, train_flag=False, inference_flag=False, packages_select_set=test_packages_select_set, input_size=input_size, compact_factor=compact_factor)
        # print(process.memory_info())
        t1 = time.time()
        op_durations["tagsets_to_matrix-testset_"+str(batch_first_idx)] = t1-t0
        op_durations["tagsets_to_matrix-testset_xsize_"+str(batch_first_idx)] = test_feature_matrix_init.shape[0]
        op_durations["tagsets_to_matrix-testset_ysize_"+str(batch_first_idx)] = test_feature_matrix_init.shape[1]
        label_matrix_list.append(test_label_matrix_init)

        # ######## Plot test feature usage as a B/W plot
        # fig, ax = plt.subplots(1, 1, figsize=(600, 10))
        # ax.imshow(test_feature_matrix_init > 0, cmap='hot', interpolation="nearest")
        # plt.savefig(cwd+'test_feature_matrix_init.pdf', format='pdf', dpi=50, bbox_inches='tight')
        # plt.close(fig)
        # gc.collect()

        # # test_feature_init_used_count = (test_feature_matrix_init > 0).sum(axis=0)
        # # fig, ax = plt.subplots(1, 1, figsize=(600, 10))
        # # ax.bar(list(range(len(test_feature_init_used_count))), test_feature_init_used_count)
        # # plt.savefig(cwd+'test_feature_init_used_count.pdf', format='pdf', dpi=50, bbox_inches='tight')
        # # plt.close(fig)
        # # gc.collect()

        # ######## Plot test label occurance as a B/W plot
        # fig, ax = plt.subplots(1, 1, figsize=(600, 10))
        # ax.imshow(test_label_matrix_init > 0, cmap='hot', interpolation="nearest")
        # plt.savefig(cwd+'test_label_matrix_init.pdf', format='pdf', dpi=50, bbox_inches='tight')
        # plt.close(fig)
        # gc.collect()

        # with open(cwd+'test_tagset_files_init.txt', 'w') as f:
        #     for line in test_tagset_files_init:
        #         f.write(f"{line}\n")
        
        # Testing
        t0 = time.time()
        pred_label_matrix_init = BOW_XGB_init.predict(test_feature_matrix_init)
        t1 = time.time()
        # pred_label_prob_matrix_init = BOW_XGB_init.predict_proba(test_feature_matrix_init)
        # t2 = time.time()
        results = one_hot_to_names(cwd+'index_label_mapping', pred_label_matrix_init)
        t3 = time.time()
        print(t1-t0, t3-t1)
        op_durations["BOW_XGB_init.predict_"+str(batch_first_idx)] = t1-t0
        # op_durations["BOW_XGB_init.predict_proba"] = t3-t2
        op_durations["one_hot_to_names_"+str(batch_first_idx)] = t3-t1
        pred_label_matrix_list.append(pred_label_matrix_init)

        # np.savetxt(cwd+'test_feature_matrix.out', test_feature_matrix, delimiter=',')

        # np.savetxt(cwd+'pred_label_matrix_init.out', pred_label_matrix_init, delimiter=',')
        # np.savetxt(cwd+'test_label_matrix_init.out', test_label_matrix_init, delimiter=',')
        # np.savetxt(cwd+'pred_label_prob_matrix_init.out', pred_label_prob_matrix_init, delimiter=',')
        # np.savetxt(cwd+'results.out', results, delimiter=',')
        # with open(cwd+'results.out', 'w') as fp:
        #     yaml.dump(results, fp)
        # with open(cwd+"pred_d_dump", 'w') as writer:
        #     results_d = {}
        #     for k,v in results.items():
        #         results_d[int(k)] = v
        #     yaml.dump(results_d, writer)
        with open(cwd+'index_label_mapping', 'rb') as fp:
            labels = np.array(pickle.load(fp))
        # labels_list.append(labels)
        print_metrics(cwd, 'metrics_init_'+str(batch_first_idx)+'.out', test_label_matrix_init, pred_label_matrix_init, labels, op_durations)
    label_matrix = np.vstack(label_matrix_list)
    pred_label_matrix = np.vstack(pred_label_matrix_list)
    # labels = np.vstack(labels_list)
    print_metrics(cwd, 'metrics_init.out', label_matrix, pred_label_matrix, labels, op_durations)

    


def run_iter_train():
    train_tags_iter_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_2/big_SL_biased_test/"
    test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_2/big_SL_biased_test/"
    cwd  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_SL/"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    # Data
    # train_tagset_files_init, train_feature_matrix_init, train_label_matrix_init = tagsets_to_matrix(train_tags_init_path, cwd=cwd, train_flag=True, inference_flag=False)
    train_tagset_files_iter, train_feature_matrix_iter, train_label_matrix_iter = tagsets_to_matrix(train_tags_iter_path, cwd=cwd, train_flag=True, inference_flag=False, iter_flag=True)
    test_tagset_files_iter, test_feature_matrix_iter, test_label_matrix_iter = tagsets_to_matrix(test_tags_path, cwd=cwd, train_flag=False, inference_flag=False, iter_flag=True)


    # Iter Training & Testing
    BOW_XGB_iter = xgb.XGBClassifier(max_depth=10, learning_rate=0.1,silent=False, objective='binary:logistic', \
                      booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                      subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1)
    # BOW_XGB_iter.load_model(cwd+'model_init.model')
    # BOW_XGB_iter.fit(train_feature_matrix_iter, train_label_matrix_iter, xgb_model=BOW_XGB_init.get_booster())
    BOW_XGB_iter.fit(train_feature_matrix_iter, train_label_matrix_iter, xgb_model=cwd+'model_init.json')
    # BOW_XGB_iter.fit(train_feature_matrix_iter, train_label_matrix_iter)
    pred_label_matrix_iter = BOW_XGB_iter.predict(test_feature_matrix_iter)
    pred_label_prob_matrix_iter = BOW_XGB_iter.predict_proba(test_feature_matrix_iter)
    pred_label_matrix_iter = pred_label_matrix_iter.add(BOW_XGB_iter.predict(test_feature_matrix_iter))
    pred_label_prob_matrix_iter = pred_label_prob_matrix_iter.add(BOW_XGB_iter.predict_proba(test_feature_matrix_iter))

    # np.savetxt(cwd+'test_feature_matrix.out', test_feature_matrix, delimiter=',')

    np.savetxt(cwd+'pred_label_matrix_iter.out', pred_label_matrix_iter, delimiter=',')
    np.savetxt(cwd+'test_label_matrix_iter.out', test_label_matrix_iter, delimiter=',')
    np.savetxt(cwd+'pred_label_prob_matrix_iter.out', pred_label_prob_matrix_iter, delimiter=',')

    with open(cwd+'index_label_mapping_iter', 'rb') as fp:
        labels = np.array(pickle.load(fp))
    print_metrics(cwd, 'metrics_iter.out', test_label_matrix_iter, pred_label_matrix_iter, labels)

def run_pred(cwd, clf_path_l, test_tags_path, n_jobs=64, n_estimators=100, packages_select_set=set(), test_batch_count=1, input_size=None, compact_factor=1, depth=1, tree_method="auto"):
    # # cwd = "/pipelines/component/cwd/"
    # cwd = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd/"
    # clf_path = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd/model_init.json"
    # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/inference_test/"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    op_durations = {}
    label_matrix_list, pred_label_matrix_list, labels_list = [], [], []
    results = defaultdict(list)
    for clf_idx, clf_path in enumerate(clf_path_l):
        with open(clf_path[:-15]+'index_label_mapping', 'rb') as fp:
            labels_list.append(np.array(pickle.load(fp)))
        t0 = time.time()
        BOW_XGB = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=depth, learning_rate=0.1,silent=False, objective='binary:logistic', \
                        booster='gbtree', n_jobs=n_jobs, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                        subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, tree_method=tree_method)
        BOW_XGB.load_model(clf_path)
        BOW_XGB.set_params(n_jobs=n_jobs)
        t1 = time.time()
        op_durations[clf_path+"\n BOW_XGB.load_model_"+str(test_batch_count)] = t1-t0
        label_matrix_list_per_clf, pred_label_matrix_list_per_clf = [],[]
        tag_files_l = [tag_file for tag_file in os.listdir(test_tags_path) if tag_file[-3:] == 'tag']
        step = len(tag_files_l)//test_batch_count+1
        for batch_first_idx in range(0, len(tag_files_l), step):
            # op_durations = {}

            # # load from previous component
            # with open(test_tags_path, 'rb') as reader:
            #     tagsets_l = pickle.load(reader)
            t0 = time.time()
            # ########### convert from tag:count strings to encoding format
            tagset_files, feature_matrix, label_matrix = tagsets_to_matrix(test_tags_path, tag_files_l = tag_files_l[batch_first_idx:batch_first_idx+step], inference_flag=False, cwd=clf_path[:-15], packages_select_set=packages_select_set, input_size=input_size, compact_factor=compact_factor) # get rid of "model_init.json" in the clf_path.
            # # ########### load a previously converted encoding format data obj
            # with open(test_tags_path+"feature_matrix.obj","rb") as filehandler:
            #     feature_matrix = pickle.load(filehandler)
            # with open(test_tags_path+"label_matrix.obj","rb") as filehandler:
            #     label_matrix = pickle.load(filehandler)
            # with open(test_tags_path+"tagset_files.obj","rb") as filehandler:
            #     tagset_files = pickle.load(filehandler)
            # # ############################################
            t1 = time.time()
            op_durations[clf_path+"\n tagsets_to_matrix-testset"+str(batch_first_idx)+"/"+str(test_batch_count)] = t1-t0
            op_durations[clf_path+"\n feature_matrix_size_0_"+str(batch_first_idx)+"/"+str(test_batch_count)] = feature_matrix.shape[0]
            op_durations[clf_path+"\n feature_matrix_size_1_"+str(batch_first_idx)+"/"+str(test_batch_count)] = feature_matrix.shape[1]
            op_durations[clf_path+"\n label_matrix_size_0_"+str(batch_first_idx)+"/"+str(test_batch_count)] = label_matrix.shape[0]
            op_durations[clf_path+"\n label_matrix_size_1_"+str(batch_first_idx)+"/"+str(test_batch_count)] = label_matrix.shape[1]
            # op_durations[clf_path+"\n tagset_files"] = tagset_files
            # ######## save train_feature_matrix_init
            # with open(cwd+"train_feature_matrix_init_"+str(clf_idx)+".mat","wb") as filehandler:
            #     np.save(filehandler, feature_matrix)

            # # debug
            # with open("/pipelines/component/cwd/tagsets.log", 'w') as writer:
            #     for tag_dict in tagsets_l:
            #         writer.write(json.dumps(tag_dict) + '\n')
            # time.sleep(5000)
            # print("labs",clf.all_labels)

            # prediction
            t0 = time.time()
            pred_label_matrix = BOW_XGB.predict(feature_matrix)
            # while True:
            #     BOW_XGB.predict(feature_matrix)
            t1 = time.time()
            results = merge_preds(results, one_hot_to_names(clf_path[:-15]+'index_label_mapping', pred_label_matrix))
            t2 = time.time()
            op_durations[clf_path+"\n BOW_XGB.predict_"+str(batch_first_idx)+"/"+str(test_batch_count)] = t1-t0
            op_durations[clf_path+"\n one_hot_to_names_"+str(batch_first_idx)+"/"+str(test_batch_count)] = t2-t1
            label_matrix_list_per_clf.append(label_matrix)
            pred_label_matrix_list_per_clf.append(pred_label_matrix)
            # op_durations_glb[clf_path] = op_durations
            # results_d = {}
            # for k,v in results.items():
            #     results_d[int(k)] = v
            print("clf"+str(clf_idx)+" pred done")
        label_matrix_list_per_clf = np.vstack(label_matrix_list_per_clf)
        pred_label_matrix_list_per_clf = np.vstack(pred_label_matrix_list_per_clf)
        label_matrix_list.append(label_matrix_list_per_clf)
        pred_label_matrix_list.append(pred_label_matrix_list_per_clf)

    Path(cwd).mkdir(parents=True, exist_ok=True)
    # with open(cwd+"pred_l_dump", 'w') as writer:
    #     # for pred in results:
    #     for pred in results.values():
    #         writer.write(f"{pred}\n")
    with open(cwd+"pred_d_dump", 'w') as writer:
        yaml.dump(results, writer)
    with open(cwd+"metrics.yaml", 'w') as writer:
        yaml.dump(op_durations, writer)
    # # ########### save the converted tag:count dict format data obj
    # with open(test_tags_path+"feature_matrix.obj","wb") as filehandler:
    #     pickle.dump(feature_matrix,filehandler)
    # with open(test_tags_path+"label_matrix.obj","wb") as filehandler:
    #     pickle.dump(label_matrix,filehandler)
    # with open(test_tags_path+"tagset_files.obj","wb") as filehandler:
    #     pickle.dump(tagset_files,filehandler)
    # # ############################################

    label_matrix = np.hstack(label_matrix_list)
    pred_label_matrix = np.hstack(pred_label_matrix_list)
    labels = np.hstack(labels_list)
    print_metrics(cwd, 'metrics_pred.out', label_matrix, pred_label_matrix, labels, op_durations)


def load_model(clf_path):
    BOW_XGB = xgb.XGBClassifier()
    BOW_XGB.load_model(clf_path)
    return BOW_XGB

if __name__ == "__main__":
    # import itertools
    # # p = psutil.Process()
    # # p.cpu_affinity([0])
    # # ###################################
    # cwd  = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_50_1_train_0shuffleidx_4testsamplebatchidx_21nsamples_32njobs_10trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par_Trueremovesharedornoisestags/"
    # BOW_XGB = load_model(cwd+"model_init.json")
    # # booster = BOW_XGB.get_booster()
    # # tree_df = booster.trees_to_dataframe()
    # # print(BOW_XGB.get_num_boosting_rounds())

    # fig, ax = plt.subplots(figsize=(30, 30))
    # xgb.plot_tree(BOW_XGB, num_trees=0, ax=ax)
    # plt.savefig(cwd+'btree0.pdf', format='pdf', dpi=500, bbox_inches='tight')
    # fig, ax = plt.subplots(figsize=(30, 30))
    # xgb.plot_importance(BOW_XGB, ax=ax)
    # plt.savefig(cwd+'feature_importance.pdf', format='pdf', dpi=500, bbox_inches='tight')
    # print()


    ##################################
    # run_init_train()
    packages_ll = {}
    n_samples_d = {}
    test_portion_d = {}
    tokenshares_filter_set_d = {}
    tokennoises_filter_set_d = {}
    # ============= data_0
    packages_l = ["wrapt", "attrs", "fsspec", "MarkupSafe", "grpcio-status", "cffi", "click", "PyJWT", "pytz", "pyasn1"]
    packages_l0 = ["psutil", "tomli", "isodate", "jsonschema", "grpcio", "soupsieve", "frozenlist", "cachetools", "botocore", "awscli"]
    packages_l1 = ["typing-extensions", "charset-normalizer", "idna", "python-dateutil", "google-api-core", "cryptography", "importlib-metadata", "s3fs", "yarl", "pyyaml"]
    packages_l2 = ["emoji", "Flask", "seaborn", "NLTK", "pytest", "zipp", "authlib", "pycparser", "colorama", "oauthlib"]
    packages_l3 = ["cmake", "nvidia-cuda-nvrtc-cu11", "jinja2", "nvidia-cuda-runtime-cu11", "wheel", "triton==2.0.0", "scikit-learn", "pandas", "dask", "deap"]
    packages_l4 = ["requests", "Scrapy", "six", "opencv-python", "simplejson", "opacus", "redis", "astropy", "biopython", "bokeh"]
    packages_l5 = ["scikit-image", "scoop", "Theano", "beautifulsoup4", "plotly", "pycaret", "mahotas", "statsmodels", "nilearn", "networkx"]
    packages_l6 = ["SQLAlchemy", "matplotlib", "scipy", "boto3", "rsa", "s3transfer", "urllib3", "setuptools", "pyspark", "pillow"]
    packages_l.extend(packages_l0)
    packages_l.extend(packages_l1)
    packages_l.extend(packages_l2)
    packages_l.extend(packages_l3)
    packages_l.extend(packages_l4)
    packages_l.extend(packages_l5)
    packages_l.extend(packages_l6)
    packages_ll["data_0"]=packages_l
    n_samples_d["data_0"]=25
    test_portion_d["data_0"]=0.2

    # ============= data_3
    packages_l = ['certifi', 'numpy', 'packaging', 'aiobotocore', 'protobuf', 'jmespath', 'googleapis-common-protos', 'platformdirs', 'google-auth', 'werkzeug', 'pydantic', 'filelock', 'pyparsing', 'async-timeout', 'aiohttp', 'docutils', 'pyarrow', 'exceptiongroup', 'pluggy', 'lxml', 'requests-oauthlib', 'tqdm', 'pyasn1-modules', 'azure-core', 'decorator', 'pyopenssl', 'greenlet', 'importlib-resources', 'multidict', 'pygments', 'websocket-client', 'pymysql', 'distlib', 'coverage', 'aiosignal', 'et-xmlfile', 'openpyxl', 'chardet', 'google-cloud-core', 'google-cloud-storage', 'asn1crypto', 'tabulate', 'google-api-python-client', 'referencing', 'iniconfig', 'tomlkit', 'rpds-py', 'paramiko', 'gitpython', 'jsonschema-specifications', 'requests-toolbelt', 'pynacl', 'more-itertools', 'proto-plus', 'psycopg2-binary', 'itsdangerous', 'azure-storage-blob', 'msal', 'google-resumable-media', 'bcrypt', 'pathspec', 'tzlocal', 'anyio', 'grpcio-tools', 'google-cloud-bigquery', 'docker', 'cython', 'mdit-py-plugins', 'joblib', 'regex', 'mypy-extensions', 'smmap', 'gitdb', 'sagemaker', 'sqlparse', 'msgpack', 'wcwidth', 'google-auth-oauthlib', 'poetry-core', 'sniffio', 'py', 'pycryptodomex', 'pyrsistent', 'azure-common', 'future', 'dnspython', 'pexpect', 'ptyprocess', 'msrest', 'jaraco-classes', 'dill', 'portalocker', 'ruamel-yaml', 'markdown', 'snowflake-connector-python', 'py4j', 'tornado', 'keyring', 'google-crc32c', 'prompt-toolkit', 'markdown-it-py', 'tenacity', 'cloudpickle', 'httplib2', 'rich', 'alembic', 'gunicorn', 'tzdata', 'awswrangler', 'fonttools', 'azure-identity', 'threadpoolctl', 'msal-extensions', 'xmltodict', 'kiwisolver', 'pycodestyle', 'termcolor', 'python-dotenv', 'tb-nightly', 'scramp', 'backoff', 'uritemplate', 'toml', 'jedi', 'webencodings', 'cachecontrol', 'marshmallow', 'poetry-plugin-export', 'ipython', 'h11', 'mccabe', 'nest-asyncio', 'cycler', 'ply', 'sortedcontainers', 'pycryptodome', 'pg8000', 'google-auth-httplib2', 'trove-classifiers', 'oscrypto', 'traitlets', 'mako', 'pyodbc', 'pkgutil-resolve-name', 'pyzmq', 'prometheus-client', 'redshift-connector', 'isort', 'toolz', 'jeepney', 'httpcore', 'secretstorage', 'adal', 'pytest-cov', 'shellingham', 'babel', 'blinker', 'datadog', 'typing-inspect', 'black', 'pymongo', 'jsonpointer', 'jupyter-client', 'defusedxml', 'google-cloud-pubsub', 'argcomplete', 'httpx', 'tensorboard', 'pyflakes', 'jupyter-core', 'sentry-sdk', 'xlrd', 'flake8', 'poetry', 'cfn-lint', 'pkginfo', 'fastapi', 'nbconvert', 'mdurl', 'requests-aws4auth', 'parso', 'asynctest', 'contourpy', 'pydantic-core', 'python-json-logger', 'absl-py', 'jsonpath-ng', 'databricks-cli', 'python-utils', 'google-cloud-bigquery-storage', 'nbformat', 'pickleshare', 'backcall', 'fastjsonschema', 'notebook', 'progressbar2', 'astroid', 'aioitertools', 'mistune', 'starlette', 'rapidfuzz', 'matplotlib-inline', 'opensearch-py', 'appdirs', 'lazy-object-proxy', 'jupyter-server', 'tensorflow', 'ipykernel', 'pbr', 'pylint', 'transformers', 'arrow', 'h5py', 'kubernetes', 'build', 'jsonpatch', 'imageio', 'setuptools-scm', 'bleach', 'huggingface-hub', 'asgiref', 'annotated-types', 'websockets', 'html5lib', 'debugpy', 'cattrs', 'pyproject-hooks', 'entrypoints', 'grpc-google-iam-v1', 'uvicorn', 'mlflow', 'smart-open', 'oauth2client', 'altair', 'msrestazure', 'multiprocess', 'numba', 'tinycss2', 'dulwich', 'llvmlite', 'tensorflow-estimator', 'zope-interface', 'lockfile', 'elasticsearch', 'mock', 'google-pasta', 'flatbuffers', 'retry', 'aiofiles', 'google-cloud-secret-manager', 'pygithub', 'mypy', 'humanfriendly', 'requests-file', 'shapely', 'orjson', 'crashtest', 'great-expectations', 'aenum', 'pysocks', 'cleo', 'comm', 'httptools', 'gast', 'querystring-parser', 'nodeenv', 'nbclient', 'tensorboard-data-server', 'contextlib2', 'identify', 'xlsxwriter', 'cached-property', 'azure-storage-file-datalake', 'croniter', 'tox', 'deepdiff', 'tokenizers', 'django', 'notebook-shim', 'send2trash', 'mysql-connector-python', 'ipywidgets', 'configparser', 'pendulum', 'execnet', 'jupyterlab-server', 'widgetsnbextension', 'text-unidecode', 'rfc3339-validator', 'overrides', 'pre-commit', 'typer', 'keras', 'json5', 'semver', 'watchdog', 'hvac', 'responses', 'torch', 'jupyterlab', 'pytzdata', 'aws-sam-translator', 'snowflake-sqlalchemy', 'python-slugify', 'cfgv', 'pipenv', 'asttokens', 'argon2-cffi', 'installer', 'dataclasses', 'sphinx', 'jupyterlab-widgets', 'executing', 'gremlinpython', 'distro', 'typeguard', 'azure-mgmt-core', 'selenium', 'jupyter-events', 'pytest-xdist', 'click-plugins', 'stack-data', 'pytest-mock', 'azure-storage-common', 'confluent-kafka', 'slack-sdk', 'pure-eval', 'opt-einsum', 'rfc3986', 'xgboost', 'tblib', 'dataclasses-json', 'opentelemetry-sdk', 'apache-airflow', 'uri-template', 'fastavro', 'tensorflow-serving-api', 'ipython-genutils', 'sentencepiece', 'futures', 'tensorflow-io-gcs-filesystem', 'sympy', 'unidecode', 'xxhash', 'safetensors', 'db-dtypes', 'pandocfilters', 'prettytable', 'patsy', 'opentelemetry-api', 'retrying', 'docopt', 'azure-mgmt-resource', 'mpmath', 'gcsfs', 'async-lru', 'jupyterlab-pygments', 'astunparse', 'setproctitle', 'terminado', 'libclang', 'pytest-runner', 'thrift', 'jsonpickle', 'semantic-version', 'ordered-set', 'azure-keyvault-secrets', 'pymssql', 'faker', 'pysftp', 'webcolors', 'argon2-cffi-bindings', 'jupyter-lsp', 'typing', 'rfc3986-validator', 'zeep', 'inflection', 'antlr4-python3-runtime', 'sphinxcontrib-serializinghtml', 'azure-datalake-store', 'graphviz', 'boto', 'fqdn', 'isoduration', 'jupyter-server-terminals', 'deprecation', 'moto', 'snowballstemmer', 'openai', 'opentelemetry-proto', 'distributed', 'azure-graphrbac', 'typed-ast', 'sphinxcontrib-htmlhelp', 'sphinxcontrib-applehelp', 'sphinxcontrib-devhelp', 'sphinxcontrib-qthelp', 'opencensus', 'ujson', 'opencensus-context', 'aioconsole', 'pathos', 'libcst', 'parsedatetime', 'stevedore', 'python-gnupg', 'google-cloud-firestore', 'pyproj', 'pandas-gbq', 'pox', 'trio', 'ppft', 'gspread', 'applicationinsights', 'numexpr', 'gevent', 'zope-event', 'kombu', 'shap', 'argparse', 'opentelemetry-exporter-otlp-proto-http', 'trio-websocket', 'google-cloud-appengine-logging', 'email-validator', 'structlog', 'loguru', 'watchtower', 'pyathena', 'torchvision', 'azure-mgmt-keyvault', 'azure-mgmt-storage', 'simple-salesforce', 'checkov', 'coloredlogs', 'apache-beam', 'tensorboard-plugin-wit', 'gsutil', 'kafka-python', 'mypy-boto3-rds', 'celery', 'pathlib2', 'pycrypto', 'wandb', 'colorlog', 'enum34', 'pybind11', 'tldextract', 'prometheus-flask-exporter', 'opentelemetry-semantic-conventions', 'types-urllib3', 'azure-cosmos', 'azure-eventhub', 'djangorestframework', 'opencensus-ext-azure', 'docstring-parser', 'lz4', 'pydata-google-auth', 'pywavelets', 'lightgbm', 'datetime', 'ecdsa', 'pyhcl', 'uamqp', 'cligj', 'google-cloud-resource-manager', 'slicer', 'fire', 'makefun', 'python-jose', 'azure-mgmt-containerregistry', 'imagesize', 'google-cloud-logging', 'keras-preprocessing', 'unittest-xml-reporting', 'alabaster', 'flask-cors', 'schema', 'hpack', 'nvidia-cudnn-cu11', 'partd', 'delta-spark', 'nvidia-cublas-cu11', 'wsproto', 'amqp', 'hypothesis', 'pytest-asyncio', 'python-http-client', 'validators', 'h2', 'azure-mgmt-authorization', 'databricks-sql-connector', 'sshtunnel', 'hyperframe', 'spacy', 'unicodecsv', 'brotli', 'fiona', 'locket', 'apache-airflow-providers-common-sql', 'holidays']
    packages_ll["data_3"]=packages_l
    n_samples_d["data_3"]=21
    test_portion_d["data_3"]=0.2

    # ============= data_4
    packages_l = ['cleo==2.0.0', 'service-identity==24.1.0', 'aws-lambda-powertools==2.34.1', 'arrow==1.2.2', 'astroid==3.0.3', 'langdetect==1.0.9', 'click-man==0.4.1', 'confection==0.1.2', 'flask-cors==3.0.9', 'matplotlib==3.8.2', 'pure-eval==0.2.0', 'types-pytz==2023.4.0.20240130', 'dateparser==1.2.0', 'markdown-it-py==2.1.0', 'feedparser==6.0.9', 'dataclasses-json==0.6.3', 'chardet==5.1.0', 'trove-classifiers==2024.2.22', 'confluent-kafka==2.2.0', 'preshed==3.0.9', 'poetry==1.8.0', 'sniffio==1.2.0', 'cinemagoer==2022.12.4', 'limits==3.9.0', 'dateparser==1.1.7', 'h2==3.2.0', 'yapf==0.40.2', 'google-cloud-firestore==2.14.0', 'tzlocal==5.0.1', 'sendgrid==6.10.0', 'python-http-client==3.3.6', 'ec2-metadata==2.13.0', 'dbt-postgres==1.7.7', 'azure-mgmt-storage==21.0.0', 'sphinx==7.2.6', 'zeep==4.1.0', 'tldextract==5.1.0', 'aiobotocore==2.11.1', 'pandas-gbq==0.20.0', 'ppft==1.7.6.8', 'google-cloud-storage==2.13.0', 'pyasn1-modules==0.2.7', 'elastic-transport==8.10.0', 'convertdate==2.3.1', 'imdbpy==2022.7.9', 'avro-python3==1.10.2', 'zeep==4.2.0', 'azure-mgmt-redhatopenshift==1.4.0', 'statsd==4.0.1', 'twine==4.0.2', 'pyopenssl==23.3.0', 'azure-identity==1.14.0', 'ddsketch==2.0.3', 'google-cloud-tasks==2.16.0', 'azure-mgmt-policyinsights==0.5.0', 'opentelemetry-exporter-otlp-proto-http==1.21.0', 'prometheus-flask-exporter==0.22.4', 'xlwt==1.2.0', 'azure-storage-file-share==12.14.1', 'networkx==3.1', 'asn1crypto==1.5.1', 'moto==5.0.2', 'mlflow==2.10.0', 'google-cloud-appengine-logging==1.4.0', 'openpyxl==3.1.0', 'google-cloud-language==2.13.2', 'mccabe==0.7.0', 'aws-requests-auth==0.4.3', 'spark-nlp==5.3.0', 'hatchling==1.21.1', 's3transfer==0.8.2', 'flask-cors==4.0.0', 'azure-datalake-store==0.0.52', 'cx-oracle==8.3.0', 'hiredis==2.3.2', 'databricks-sdk==0.20.0', 'pydub==0.25.0', 'flask-appbuilder==4.4.1', 'httpx==0.25.2', 'flask-login==0.6.2', 'databricks-sql-connector==3.0.3', 'azure-mgmt-storage==21.1.0', 'google-cloud-resource-manager==1.12.2', 'rfc3339-validator==0.1.2', 'openapi-spec-validator==0.7.1', 'tensorflow-text==2.14.0', 'msgpack==1.0.6', 'click-didyoumean==0.1.0', 'hvac==2.1.0', 'flake8==6.1.0', 'elasticsearch==8.12.1', 'azure-mgmt-compute==30.5.0', 'resolvelib==1.0.1', 'gunicorn==21.2.0', 'fiona==1.9.3', 'cattrs==23.2.1', 'blinker==1.7.0', 'cdk-nag==2.28.45', 'python-daemon==2.3.0', 'pyjwt==2.7.0', 'sh==2.0.4', 'hologram==0.0.14', 'enum34==1.1.9', 'nvidia-nvjitlink-cu12==12.2.140', 'azure-common==1.1.26', 'grpcio-tools==1.62.0', 'greenlet==3.0.3', 'timm==0.9.11', 'aiohttp==3.9.3', 'rdflib==6.3.1', 'seaborn==0.13.1', 'gspread==6.0.0', 'python-json-logger==2.0.6', 'jsonpatch==1.31', 'graphviz==0.20', 'azure-mgmt-consumption==9.0.0', 'pipx==1.4.2', 'hvac==1.2.1', 'xlrd==1.2.0', 'aiodns==3.0.0', 'opentelemetry-api==1.22.0', 'firebase-admin==6.3.0', 'awscli==1.32.47', 'notebook==7.0.8', 'parso==0.8.2', 'tensorflow-hub==0.16.1', 'flask-limiter==3.5.0', 'marshmallow-dataclass==8.6.0', 'azure-mgmt-web==7.2.0', 'pynacl==1.4.0', 'requests-aws4auth==1.2.2', 'rpds-py==0.17.1', 'graphviz==0.20.1', 'geoip2==4.6.0', 'torchvision==0.17.0', 'google-cloud-spanner==3.40.1', 'dm-tree==0.1.8', 'responses==0.24.0', 'boto3==1.34.48', 'jsonpatch==1.33', 'django-extensions==3.2.0', 'aws-lambda-powertools==2.33.1', 'cdk-nag==2.28.52', 'future==0.18.2', 'opt-einsum==3.3.0', 'statsd==4.0.0', 'azure-mgmt-netapp==11.0.0', 'html5lib==1.0.1', 'pydash==7.0.6', 'langcodes==3.2.1', 'voluptuous==0.14.1', 'flask-limiter==3.4.1', 'opencv-python-headless==4.9.0.80', 'attrs==23.1.0', 'dbt-postgres==1.7.9', 'user-agents==2.1', 'build==0.10.0', 'markdown-it-py==2.2.0', 'boto3-stubs==1.34.55', 'babel==2.13.1', 'types-pyyaml==6.0.12.10', 'ipaddress==1.0.23', 'pytest-rerunfailures==12.0', 'zope-event==4.5.0', 'flask-jwt-extended==4.6.0', 'watchfiles==0.20.0', 'gitdb==4.0.9', 'cleo==2.0.1', 'gql==3.4.1', 'phonenumbers==8.13.28', 'imbalanced-learn==0.11.0', 'webdriver-manager==4.0.0', 'datasets==2.17.1', 'phonenumbers==8.13.29', 'rich==13.6.0', 'starlette==0.37.1', 'azure-mgmt-iotcentral==9.0.0', 'twilio==8.12.0', 'rich==13.7.0', 'pypandoc==1.11', 'addict==2.2.1', 'applicationinsights==0.11.9', 'alabaster==0.7.16', 'zstandard==0.20.0', 'celery==5.3.4', 'comm==0.1.4', 'google-cloud-build==3.23.1', 'typer==0.8.0', 'safetensors==0.4.2', 'nbclient==0.7.4', 'time-machine==2.11.0', 'funcsigs==1.0.2', 'azure-eventgrid==4.17.0', 'send2trash==1.7.1', 'nvidia-nvjitlink-cu12==12.3.52', 'universal-pathlib==0.2.0', 'pycparser==2.20', 'azure-mgmt-apimanagement==2.1.0', 'google-re2==1.0', 'orbax-checkpoint==0.5.2', 'nh3==0.2.14', 'mypy-boto3-appflow==1.33.0', 'pymeeus==0.5.10', 'types-protobuf==4.24.0.4', 'pytest-cov==3.0.0', 'asyncache==0.2.0', 'python-jose==3.1.0', 'google-cloud-vision==3.6.0', 'boltons==23.1.1', 'holidays==0.42', 'websockets==11.0.3', 'httptools==0.5.0', 'zstandard==0.22.0', 'slack-sdk==3.27.1', 'tinycss2==1.2.1', 'flit-core==3.9.0', 'configparser==6.0.0', 'marshmallow==3.20.1', 'datasets==2.18.0', 'time-machine==2.12.0', 'pycryptodomex==3.19.0', 'nltk==3.7', 'boto3-stubs==1.34.49', 'autopep8==2.0.3', 'trove-classifiers==2024.1.31', 'tensorflow==2.15.0', 'datadog-api-client==2.20.0', 'blis==0.7.11', 'python-utils==3.8.2', 'azure-graphrbac==0.60.0', 'pyspnego==0.9.1', 'nvidia-cublas-cu12==12.3.2.9', 'natsort==8.3.1', 'kiwisolver==1.4.4', 'jedi==0.19.1', 'slack-sdk==3.26.2', 'flask-wtf==1.2.0', 'nvidia-nccl-cu12==2.18.3', 'googleapis-common-protos==1.60.0', 'dpath==2.1.6', 'ruamel-yaml==0.18.5', 'pyserial==3.3', 'smdebug-rulesconfig==1.0.1', 'monotonic==1.4', 'avro-python3==1.10.1', 'cloudpickle==2.2.0', 'oldest-supported-numpy==2023.12.21', 'jupyterlab-widgets==3.0.8', 'pyproj==3.6.0', 'aws-sam-translator==1.85.0', 'pygithub==2.1.1', 'google-cloud-container==2.39.0', 'ansible-core==2.15.9', 'azure-storage-common==2.1.0', 'jupyterlab-widgets==3.0.9', 'flake8==7.0.0', 'dataclasses==0.5', 'dataclasses-json==0.6.2', 'azure-mgmt-synapse==2.0.0', 'tensorflow-datasets==4.9.2', 'google-cloud-pubsub==2.19.5', 'spark-nlp==5.3.1', 'black==24.1.0', 'azure-mgmt-servicebus==8.0.0', 'colorama==0.4.5', 'jsonpath-ng==1.5.3', 'docutils==0.19', 'pathlib==1.0.1', 'mlflow==2.10.2', 'h11==0.12.0', 'altair==5.1.1', 'jsonpath-ng==1.6.1', 'pytest-asyncio==0.23.5', 'grpc-google-iam-v1==0.12.7', 'tokenizers==0.15.1', 'numba==0.59.0', 'fabric==3.2.0', 'kfp-server-api==2.0.4', 'requests-toolbelt==0.10.0', 'nvidia-cufft-cu12==11.0.12.1', 'azure-mgmt-servicefabric==2.1.0', 'google-cloud-datacatalog==3.18.2', 'azure-mgmt-signalr==1.0.0', 'azure-mgmt-reservations==2.3.0', 'thinc==8.2.1', 'phonenumbers==8.13.30', 'azure-mgmt-netapp==10.0.0', 'azure-synapse-artifacts==0.16.0', 'argparse==1.3.0', 'gsutil==5.25', 'sqlalchemy==2.0.28', 'apache-airflow-providers-common-sql==1.11.0', 'patsy==0.5.4', 'sendgrid==6.11.0', 'pydash==7.0.5', 'aenum==3.1.14', 'pika==1.3.2', 'distlib==0.3.8', 'orjson==3.9.14', 'typed-ast==1.5.4', 'python-dotenv==1.0.0', 'comm==0.2.1', 'google-cloud-monitoring==2.19.2', 'user-agents==2.2.0', 'rpds-py==0.18.0', 'requests-file==2.0.0', 'stringcase==1.0.4', 'azure-mgmt-rdbms==10.1.0', 'gql==3.5.0', 'stack-data==0.6.2', 'gunicorn==21.1.0', 'lxml==5.1.0', 'jupyter-client==8.4.0', 'scramp==1.4.2', 'frozenlist==1.4.0', 'fasteners==0.17.3', 'jaxlib==0.4.24', 'pandocfilters==1.4.3', 'aiobotocore==2.12.1', 'ldap3==2.9.1', 'pyathena==3.2.1', 'mmh3==4.0.1', 'google-cloud-storage==2.12.0', 'types-setuptools==69.1.0.20240302', 'werkzeug==2.3.8', 'email-validator==2.1.0', 'langcodes==3.2.0', 'sqlalchemy==2.0.25', 'pandocfilters==1.5.0', 'oldest-supported-numpy==2023.12.12', 'azure-mgmt-iothubprovisioningservices==1.0.0', 'google-auth-httplib2==0.1.0', 'jsonschema-specifications==2023.11.1', 'jsonschema-specifications==2023.12.1', 'databricks-cli==0.17.7', 'apache-airflow-providers-amazon==8.16.0', 'python-json-logger==2.0.7', 'pylint==3.0.4', 'gast==0.5.4', 'cdk-nag==2.28.46', 'querystring-parser==1.2.4', 'mypy-boto3-s3==1.33.2', 'azure-mgmt-redis==14.1.0', 'pytz==2023.4', 'azure-servicebus==7.11.3', 'netaddr==0.10.1', 'telethon==1.33.1', 'pyathena==3.2.0', 'pexpect==4.8.0', 'atomicwrites==1.3.0', 'selenium==4.18.0', 'google-api-core==2.17.0', 'iniconfig==1.1.1', 'voluptuous==0.14.2', 'orbax-checkpoint==0.5.3', 'opentelemetry-exporter-otlp==1.21.0', 'azure-mgmt-sql==3.0.0', 'configargparse==1.5.5', 'async-timeout==4.0.1', 'google-cloud-bigquery==3.17.1', 'azure-mgmt-datalake-analytics==0.5.0', 'pymsteams==0.1.16', 'notebook==7.0.7', 'pure-eval==0.2.2', 'lightning-utilities==0.10.0', 'tenacity==8.2.2', 'openapi-spec-validator==0.7.0', 'pyaml==23.12.0', 'pre-commit==3.6.0', 'dbt-core==1.7.8', 'grpcio-status==1.62.0', 'google-cloud-kms==2.21.2', 'dpath==2.1.5', 'aws-requests-auth==0.4.1', 'dbt-postgres==1.7.8', 'email-validator==2.0.0', 'ftfy==6.1.0', 'simple-salesforce==1.12.5', 'coverage==7.4.2', 'h2==3.1.1', 'wasabi==0.10.1', 'parameterized==0.7.5', 'sh==2.0.5', 'pox==0.3.4', 'azure-mgmt-imagebuilder==1.1.0', 'boto3==1.34.47', 'cramjam==2.8.1', 'py==1.10.0', 'pickleshare==0.7.3', 'ordered-set==4.1.0', 'google-cloud-pubsub==2.19.7', 'outcome==1.3.0', 'geographiclib==1.52', 'sagemaker==2.210.0', 'pypandoc==1.13', 'azure-mgmt-msi==6.1.0', 'flask==3.0.1', 'itsdangerous==2.1.1', 'pluggy==1.4.0', 'progressbar2==4.4.1', 'apache-airflow-providers-snowflake==5.2.1', 'pyasn1==0.4.8', 'jeepney==0.8.0', 'pyopenssl==23.2.0', 'jaydebeapi==1.2.3', 'user-agent==0.1.9', 'omegaconf==2.2.2', 'gevent==24.2.1', 'deepdiff==6.7.0', 'spacy-legacy==3.0.10', 'murmurhash==1.0.9', 'executing==1.2.0', 'asgiref==3.7.2', 'py-cpuinfo==8.0.0', 'zope-event==4.6', 'twine==5.0.0', 'async-timeout==4.0.2', 'maxminddb==2.5.2', 'googleapis-common-protos==1.61.0', 'google-cloud-datastore==2.18.0', 'azure-mgmt-hdinsight==9.0.0', 'django-cors-headers==4.2.0', 'psycopg==3.1.17', 'levenshtein==0.25.0', 'databricks-cli==0.18.0', 'jaraco-classes==3.2.3', 'azure-mgmt-applicationinsights==4.0.0', 'tblib==2.0.0', 'linkify-it-py==2.0.2', 'universal-pathlib==0.2.2', 'astroid==3.1.0', 'editables==0.5', 'python-jsonpath==0.10.3', 'keras-applications==1.0.6', 'html5lib==0.999999999', 'billiard==4.2.0', 'google-cloud-monitoring==2.19.0', 'tomlkit==0.12.2', 'overrides==7.7.0', 'pytz==2023.3', 'korean-lunar-calendar==0.2.1', 'future==0.18.3', 'apache-airflow-providers-databricks==6.2.0', 'openai==1.13.3', 'botocore==1.34.54', 'azure-mgmt-devtestlabs==3.0.0', 'nvidia-curand-cu12==10.3.4.101', 'docutils==0.20.1', 'h3==3.7.3', 'mypy-extensions==0.4.3', 'pathlib==1.0', 'distributed==2024.1.1', 'certifi==2023.11.17', 'pygithub==1.59.1', 'mypy-boto3-redshift-data==1.29.0', 'matplotlib-inline==0.1.5', 'scp==0.14.3', 'azure-kusto-ingest==4.3.1', 'azure-storage-queue==12.8.0', 'scramp==1.4.4', 'cycler==0.11.0', 'wtforms==3.1.2', 'deprecation==2.0.6', 'backoff==2.2.0', 'yamllint==1.34.0', 'azure-keyvault-administration==4.3.0', 'google-cloud-compute==1.17.0', 'prettytable==3.8.0', 'opencensus-ext-azure==1.1.11', 'lark==1.1.9', 'setuptools==69.1.0', 'billiard==4.1.0', 'asynctest==0.12.4', 'azure-mgmt-redhatopenshift==1.2.0', 'chardet==5.2.0', 'async-lru==2.0.2', 'nbclassic==0.5.6', 'amqp==5.1.1', 'pytest-metadata==3.1.1', 'geoip2==4.7.0', 'fastparquet==2024.2.0', 'wasabi==1.1.1', 'azure-mgmt-authorization==4.0.0', 'scp==0.14.4', 'altair==5.2.0', 'ruff==0.2.2', 'pydata-google-auth==1.8.1', 'ua-parser==0.16.1', 'pyproj==3.5.0', 'sphinxcontrib-htmlhelp==2.0.5', 'shortuuid==1.0.10', 'parse-type==0.6.2', 'tensorflow-datasets==4.9.3', 'pytest-mock==3.12.0', 'outcome==1.2.0', 'docopt==0.6.1', 'setuptools-scm==8.0.3', 'cfn-lint==0.85.0', 'opentelemetry-exporter-otlp-proto-http==1.23.0', 'faker==23.2.0', 'azure-mgmt-eventhub==10.1.0', 'types-requests==2.31.0.20240106', 'jaxlib==0.4.22', 'polars==0.20.9', 'oauth2client==4.1.2', 'bokeh==3.3.2', 'grpcio-health-checking==1.60.0', 'dm-tree==0.1.7', 'statsmodels==0.14.0', 'nvidia-nvtx-cu12==12.2.140', 'knack==0.11.0', 'virtualenv==20.25.0', 'azure-cosmos==4.4.0', 'contourpy==1.1.1', 'nvidia-cusolver-cu12==11.5.3.52', 'contourpy==1.1.0', 'imagesize==1.4.0', 'apache-airflow-providers-amazon==8.17.0', 'langsmith==0.1.18', 'jupyterlab==4.1.1', 'azure-appconfiguration==1.3.0', 'transformers==4.38.2', 'promise==2.3', 'pydantic-core==2.16.3', 'faker==24.0.0', 'azure-mgmt-signalr==1.2.0', 'toolz==0.12.1', 'parse-type==0.6.1', 's3fs==2023.12.1', 'shapely==2.0.2', 'jupyterlab-server==2.25.3', 'databricks-cli==0.17.8', 'cdk-nag==2.28.53', 'setproctitle==1.3.2', 'azure-mgmt-datalake-store==0.5.0', 'h3==3.7.6', 'pyyaml==6.0', 'azure-servicebus==7.11.4', 'isort==5.13.1', 'xyzservices==2023.7.0', 'django==4.2.10', 'ml-dtypes==0.3.1', 'typer==0.9.0', 'nvidia-curand-cu12==10.3.4.107', 'apache-airflow-providers-imap==3.4.0', 'aiodns==3.1.0', 'google-cloud-vision==3.7.0', 'docstring-parser==0.14', 'knack==0.10.0', 'types-requests==2.31.0.20240218', 'gradio==4.19.0', 'grpcio==1.60.0', 'msal-extensions==1.1.0', 'databricks-sql-connector==3.0.2', 'apache-beam==2.53.0', 'jupyter-console==6.6.2', 'humanfriendly==10.0', 'jupyter-client==8.3.1', 'cmdstanpy==1.2.1', 'kr8s==0.13.3', 'croniter==2.0.2', 'google-cloud-kms==2.21.0', 'freezegun==1.3.0', 'fastjsonschema==2.19.1', 'ansible-core==2.15.8', 'elasticsearch-dsl==8.9.0', 'retrying==1.3.4', 'entrypoints==0.3', 'deprecation==2.0.7', 'botocore-stubs==1.34.49', 'azure-mgmt-security==4.0.0', 'oauth2client==4.1.1', 'ftfy==6.1.3', 'smart-open==6.4.0', 'aiofiles==22.1.0', 'azure-mgmt-iotcentral==4.0.0', 'platformdirs==4.1.0', 'absl-py==2.0.0', 'google-cloud-dataproc==5.9.0', 'docopt==0.6.0', 'matplotlib==3.8.1', 'attrs==23.2.0', 'dataclasses==0.6', 'ipaddress==1.0.22', 'setuptools-rust==1.8.0', 'types-s3transfer==0.8.2', 'executing==2.0.1', 'tox==4.13.0', 'jsondiff==2.0.0', 'click==8.1.6', 'poetry-core==1.9.0', 'nvidia-cuda-nvrtc-cu12==12.3.103', 'flask-wtf==1.2.1', 'omegaconf==2.3.0', 'unicodecsv==0.13.0', 'commonmark==0.9.0', 'slackclient==2.9.3', 'xarray==2024.1.1', 'httpx==0.26.0', 'websockets==12.0', 'nh3==0.2.15', 'iniconfig==2.0.0', 'python-gnupg==0.5.1', 'botocore==1.34.53', 'kombu==5.3.5', 'boto3==1.34.49', 'zipp==3.16.2', 'marshmallow-enum==1.4.1', 'cron-descriptor==1.4.0', 'tokenizers==0.15.0', 'pytest-rerunfailures==11.1.2', 'lark==1.1.8', 'google-cloud-spanner==3.42.0', 'imbalanced-learn==0.10.1', 'pytorch-lightning==2.2.1', 'retrying==1.3.2', 'fastavro==1.9.3', 'tensorflow-hub==0.16.0', 'botocore==1.34.49', 'redis==5.0.0', 'mkdocs-material==9.5.10', 'oldest-supported-numpy==2023.10.25', 'sphinxcontrib-qthelp==1.0.5', 'h11==0.14.0', 'nh3==0.2.13', 'sqlparse==0.4.2', 'types-urllib3==1.26.25.13', 'google-cloud-tasks==2.16.2', 'pygments==2.17.1', 'azure-mgmt-containerservice==28.0.0', 'azure-batch==13.0.0', 'markdown==3.5', 'loguru==0.7.1', 'inflection==0.5.0', 'torchvision==0.17.1', 'psycopg2-binary==2.9.7', 'geopy==2.4.1', 'tensorboard-data-server==0.7.0', 'azure-mgmt-search==8.0.0', 'jsonschema==4.20.0', 'vine==5.1.0', 'prometheus-flask-exporter==0.22.3', 'requests-toolbelt==1.0.0', 'datadog==0.48.0', 'tinycss2==1.1.1', 'pyasn1-modules==0.3.0', 'google-cloud-bigquery==3.17.0', 'azure-mgmt-containerregistry==10.1.0', 'langchain==0.1.9', 'zope-interface==6.2', 'uvloop==0.17.0', 'azure-cli-telemetry==1.0.7', 'pypdf2==2.12.1', 'dbt-extractor==0.4.1', 'colorama==0.4.6', 'catalogue==2.0.10', 'aws-xray-sdk==2.11.0', 'xxhash==3.2.0', 'packaging==23.2', 'multiprocess==0.70.15', 'safetensors==0.4.0', 'jax==0.4.23', 'toolz==0.11.2', 'pydeequ==1.1.1', 'distlib==0.3.6', 'alabaster==0.7.14', 'google-cloud-bigtable==2.21.0', 'msal==1.26.0', 'sphinxcontrib-devhelp==1.0.4', 'fonttools==4.47.2', 'lockfile==0.12.2', 'onnxruntime==1.16.3', 'redshift-connector==2.0.917', 'pycodestyle==2.11.1', 'srsly==2.4.7', 'typer==0.7.0', 'envier==0.4.0', 'pytz==2024.1', 'grpc-google-iam-v1==0.13.0', 'bandit==1.7.6', 'httptools==0.6.0', 'uc-micro-py==1.0.1', 'boto==2.49.0', 'libclang==16.0.0', 'fiona==1.9.5', 'widgetsnbextension==4.0.8', 'pbr==5.11.0', 'python-http-client==3.3.5', 'nvidia-cublas-cu12==12.3.4.1', 'fastapi==0.110.0', 'sortedcontainers==2.4.0', 'terminado==0.17.0', 'jeepney==0.7.0', 'aws-xray-sdk==2.12.1', 'autopep8==2.0.2', 'torchmetrics==1.3.0', 'apache-airflow-providers-sqlite==3.6.0', 'google-cloud-datastore==2.17.0', 'pytzdata==2019.2', 'tensorflow-serving-api==2.14.1', 'msrestazure==0.6.3', 'openpyxl==3.1.2', 'boto3-stubs==1.34.54', 'types-python-dateutil==2.8.19.13', 'importlib-metadata==7.0.0', 'distrax==0.1.3', 'python-docx==1.0.0', 'xlsxwriter==3.1.8', 'jsonpointer==2.4', 'django-cors-headers==4.3.1', 'azure-mgmt-datalake-store==0.4.0', 'jupyter-events==0.9.0', 'user-agent==0.1.10', 'gast==0.5.3', 'ply==3.11', 'natsort==8.3.0', 'azure-mgmt-rdbms==9.1.0', 'looker-sdk==23.20.1', 'azure-mgmt-cognitiveservices==13.3.0', 'ppft==1.7.6.6', 'pymongo==4.6.1', 'setuptools==69.0.3', 'annotated-types==0.6.0', 'aiohttp==3.9.2', 'cramjam==2.8.2', 'pysocks==1.7.1', 'junit-xml==1.7', 'pymsteams==0.2.2', 'uvloop==0.19.0', 'iso8601==2.0.0', 'sphinxcontrib-qthelp==1.0.6', 'typing==3.7.4.1', 'azure-mgmt-keyvault==10.2.3', 'chex==0.1.83', 'google-cloud-appengine-logging==1.4.2', 'pkginfo==1.9.4', 'shortuuid==1.0.12', 'azure-mgmt-dns==8.1.0', 'pyrsistent==0.19.2', 'marshmallow-sqlalchemy==0.29.0', 'thinc==8.2.2', 'beautifulsoup4==4.12.3', 'hatchling==1.20.0', 'databricks-sdk==0.19.1', 'unicodecsv==0.14.0', 'azure-mgmt-media==10.1.0', 'azure-mgmt-sql==3.0.1', 'flask-caching==2.0.1', 'frozendict==2.3.10', 'loguru==0.7.0', 'azure-mgmt-network==25.2.0', 'pycodestyle==2.11.0', 'twilio==8.11.1', 'pydantic==2.6.0', 'hypothesis==6.98.15', 'google-cloud-bigquery==3.17.2', 'jupyter-server-terminals==0.5.0', 'apache-airflow-providers-slack==8.5.1', 'faker==23.3.0', 'azure-mgmt-devtestlabs==9.0.0', 'keyring==24.1.1', 'uc-micro-py==1.0.3', 'numpy==1.26.2', 'google-cloud-build==3.23.0', 'apispec==6.3.1', 'sphinx-rtd-theme==1.3.0', 'incremental==22.10.0', 'torchvision==0.16.2', 'cymem==2.0.6', 'cog==0.9.3', 'fsspec==2023.12.2', 'tokenizers==0.15.2', 'cx-oracle==8.2.0', 'twilio==8.13.0', 'fuzzywuzzy==0.18.0', 'text-unidecode==1.2', 'ruamel-yaml-clib==0.2.6', 'delta-spark==3.1.0', 'gsutil==5.26', 'aws-lambda-powertools==2.34.0', 'wrapt==1.14.1', 'apache-airflow==2.8.2', 'black==24.1.1', 'azure-mgmt-recoveryservicesbackup==8.0.0', 'psutil==5.9.6', 'webdriver-manager==3.9.1', 'azure-mgmt-redis==14.2.0', 'types-awscrt==0.20.3', 'pandas==2.2.0', 'pipenv==2023.11.17', 'defusedxml==0.7.1', 'setuptools==69.1.1', 'azure-mgmt-batchai==1.0.0', 'py==1.11.0', 'office365-rest-python-client==2.5.4', 'llvmlite==0.41.0', 'azure-eventhub==5.11.4', 'retry==0.8.1', 'pydot==2.0.0', 'flask-login==0.6.1', 'httplib2==0.22.0', 'click==8.1.5', 'click-man==0.4.0', 'azure-mgmt-search==9.1.0', 'openai==1.11.1', 'lightgbm==4.2.0', 'types-protobuf==4.24.0.20240129', 'azure-mgmt-monitor==6.0.0', 'ruff==0.2.0', 'json5==0.9.15', 'dvc-render==1.0.1', 'apache-airflow-providers-ftp==3.6.0', 'snowflake-connector-python==3.7.1', 'keras==3.0.5', 'fqdn==1.5.1', 'virtualenv==20.24.7', 'sshtunnel==0.3.2', 'pip-tools==7.2.0', 'azure-mgmt-sql==2.1.0', 'google-api-python-client==2.118.0', 'flask==3.0.2', 'evergreen-py==3.6.20', 'great-expectations==0.18.9', 'tensorboard-plugin-wit==1.8.1', 'distro==1.9.0', 'hdfs==2.7.3', 'rsa==4.9', 'typing-inspect==0.8.0', 'apache-airflow-providers-http==4.9.0', 'notebook-shim==0.2.3', 'pyotp==2.7.0', 'azure-kusto-ingest==4.2.0', 'azure-mgmt-media==10.0.0', 'poetry-plugin-export==1.4.0', 'kfp-pipeline-spec==0.2.1', 'levenshtein==0.23.0', 'deprecated==1.2.13', 'redis==4.6.0', 'nvidia-cuda-cupti-cu12==12.3.101', 'lz4==4.3.1', 'evidently==0.4.16', 'makefun==1.15.0', 'stevedore==5.0.0', 'botocore-stubs==1.34.47', 'pyaml==23.9.7', 'click-didyoumean==0.2.0', 'apache-airflow-providers-ftp==3.7.0', 'dvclive==3.43.0', 'pgpy==0.6.0', 'nest-asyncio==1.6.0', 'types-s3transfer==0.10.0', 'nltk==3.8.1', 'yapf==0.40.1', 'frozenlist==1.4.1', 'wsproto==1.2.0', 'botocore==1.34.55', 'configobj==5.0.6', 'requests-aws4auth==1.2.3', 'pymssql==2.2.11', 'distro==1.8.0', 'tornado==6.3.2', 'wsproto==1.1.0', 'openpyxl==3.1.1', 'azure-mgmt-imagebuilder==1.2.0', 'sympy==1.12', 'pika==1.3.1', 'coloredlogs==14.3', 'ansible==8.6.1', 'numba==0.58.0', 'azure-mgmt-hdinsight==8.0.0', 'marshmallow-enum==1.4', 'pycryptodomex==3.20.0', 'azure-mgmt-datafactory==3.1.0', 'pytimeparse==1.1.6', 'azure-kusto-data==4.3.0', 'pytest-cov==4.1.0', 'mypy-boto3-redshift-data==1.34.0', 'importlib-resources==6.1.1', 'voluptuous==0.13.1', 'certifi==2024.2.2', 'colorlog==6.8.0', 'fastparquet==2023.10.1', 'argcomplete==3.2.1', 'apache-beam==2.52.0', 'zope-interface==6.1', 'pycares==4.4.0', 'mkdocs-material==9.5.12', 'azure-keyvault-secrets==4.7.0', 'azure-storage-file-datalake==12.13.2', 'kfp-pipeline-spec==0.2.2', 'gremlinpython==3.7.0', 'cfn-lint==0.86.0', 'jinja2==3.1.1', 'korean-lunar-calendar==0.3.1', 'python-gitlab==4.4.0', 'asyncio==3.4.1', 'cookiecutter==2.4.0', 'configupdater==3.1.1', 'prettytable==3.9.0', 'torchmetrics==1.2.1', 'hiredis==2.2.2', 'azure-cli==2.58.0', 'asyncache==0.3.0', 'xlwt==1.1.2', 'jax==0.4.24', 'onnxruntime==1.17.1', 'tensorboard-plugin-wit==1.8.0', 'cramjam==2.8.0', 'watchdog==4.0.0', 'google-cloud-dlp==3.15.1', 'jupyter-server==2.13.0', 'newrelic==9.5.0', 'azure-datalake-store==0.0.53', 'lockfile==0.11.0', 'checkov==3.2.23', 'kiwisolver==1.4.3', 'cfgv==3.4.0', 'azure-mgmt-consumption==10.0.0', 'ipython==8.18.0', 'execnet==2.0.0', 'jupyterlab-pygments==0.2.2', 'jupyterlab==4.1.2', 'pysftp==0.2.7', 'docutils==0.20', 'azure-mgmt-marketplaceordering==0.2.1', 'firebase-admin==6.4.0', 'tox==4.12.1', 'kubernetes==29.0.0', 'jupyter-server==2.12.5', 'requests-mock==1.9.3', 'slicer==0.0.5', 'pyelftools==0.28', 'imdbpy==2020.9.25', 'google-cloud-audit-log==0.2.4', 'wasabi==1.1.2', 'django-filter==23.5', 'identify==2.5.34', 'types-urllib3==1.26.25.12', 'pyyaml==6.0.1', 'azure-storage-blob==12.19.0', 'gradio==4.19.1', 'pytzdata==2020.1', 'azure-mgmt-eventhub==11.0.0', 'validators==0.22.0', 'addict==2.3.0', 'poetry==1.7.1', 'jupyter-lsp==2.2.2', 'pyathena==3.3.0', 'ijson==3.2.2', 'python-magic==0.4.27', 'tinycss2==1.2.0', 'statsd==3.3.0', 'yamllint==1.35.0', 'packaging==23.0', 'gitpython==3.1.40', 'python-dotenv==1.0.1', 'azure-cosmos==4.5.0', 'argon2-cffi==23.1.0', 'poetry-core==1.8.1', 'google-resumable-media==2.7.0', 'psycopg==3.1.16', 'amqp==5.1.0', 'leather==0.4.0', 'parse==1.19.1', 'ansible==8.7.0', 'keras==3.0.4', 'tensorflow==2.14.1', 'opentelemetry-exporter-otlp==1.22.0', 'apache-airflow-providers-ssh==3.10.0', 'bcrypt==4.0.1', 'google-cloud-monitoring==2.19.1', 'pycares==4.3.0', 'dvc-render==1.0.0', 'keras-preprocessing==1.1.2', 'azure-mgmt-containerregistry==10.3.0', 'cmdstanpy==1.2.0', 'pathlib2==2.3.5', 'jax==0.4.25', 'pickleshare==0.7.5', 'pymssql==2.2.10', 'azure-mgmt-reservations==2.1.0', 'azure-synapse-artifacts==0.17.0', 'pathspec==0.12.1', 'tensorflow-hub==0.15.0', 'apache-airflow-providers-ftp==3.6.1', 'ray==2.9.1', 'soupsieve==2.5', 'click-plugins==1.1.1', 'shellingham==1.5.4', 'keras==3.0.3', 'google-cloud-resource-manager==1.12.0', 'langchain==0.1.7', 'py4j==0.10.9.7', 'readme-renderer==40.0', 'alabaster==0.7.15', 'google-resumable-media==2.6.0', 'db-dtypes==1.1.1', 'tensorboard==2.16.0', 'texttable==1.6.7', 'azure-mgmt-appconfiguration==2.2.0', 'tensorboard-plugin-wit==1.7.0', 'pytorch-lightning==2.1.4', 'platformdirs==4.2.0', 'azure-mgmt-recoveryservices==2.5.0', 'aiofiles==23.1.0', 'google-cloud-logging==3.9.0', 'adal==1.2.5', 'blis==0.9.0', 'contourpy==1.2.0', 'async-generator==1.9', 'fastavro==1.9.2', 'types-redis==4.6.0.20240218', 'markdown==3.5.2', 'pre-commit==3.6.1', 'awswrangler==3.5.1', 'tensorflow-io==0.34.0', 'azure-mgmt-containerinstance==9.2.0', 'types-urllib3==1.26.25.14', 'jaraco-classes==3.3.0', 'ray==2.9.3', 'jira==3.5.1', 'mypy==1.7.1', 'proto-plus==1.22.3', 'geographiclib==1.50', 'tomli==2.0.1', 'opentelemetry-proto==1.22.0', 'pygithub==2.2.0', 'azure-storage-common==2.0.0', 'flatbuffers==23.5.8', 'beautifulsoup4==4.12.2', 'boltons==23.1.0', 'scp==0.14.5', 'tensorflow-io==0.36.0', 'vine==5.0.0', 'azure-mgmt-core==1.3.1', 'hyperframe==6.0.1', 'matplotlib==3.8.3', 'kombu==5.3.4', 'aws-psycopg2==1.2.0', 'python-box==7.1.1', 'smdebug-rulesconfig==0.1.7', 'jsonschema-specifications==2023.11.2', 'itsdangerous==2.1.2', 'azure-core==1.30.1', 'pip==23.3.1', 'azure-mgmt-servicebus==8.1.0', 'tensorflow==2.14.0', 'incremental==17.5.0', 'inflect==7.0.0', 'holidays==0.41', 'flax==0.8.1', 'apispec==6.3.0', 'cached-property==1.5.1', 'parsedatetime==2.5', 'hyperlink==21.0.0', 'stevedore==5.1.0', 'cloudpathlib==0.18.1', 'timm==0.9.12', 'decorator==5.0.9', 'croniter==2.0.1', 'nvidia-cuda-runtime-cu12==12.3.52', 'validators==0.21.2', 'bytecode==0.15.1', 'build==1.1.1', 'soupsieve==2.4', 'shellingham==1.5.2', 'structlog==24.1.0', 'cattrs==23.2.3', 'python-jsonpath==0.10.2', 'amqp==5.2.0', 'langdetect==1.0.8', 'opentelemetry-proto==1.23.0', 'snowballstemmer==2.1.0', 'ultralytics==8.1.18', 'typing-extensions==4.9.0', 'simple-salesforce==1.12.4', 'scikit-image==0.21.0', 'astunparse==1.6.1', 'exceptiongroup==1.1.2', 'debugpy==1.7.0', 'checkov==3.2.29', 'ijson==3.2.1', 'hyperframe==5.2.0', 'kafka-python==2.0.2', 'nvidia-cudnn-cu12==8.9.7.29', 'azure-mgmt-cosmosdb==9.2.0', 'seaborn==0.13.2', 'envier==0.5.0', 'iniconfig==1.1.0', 'magicattr==0.1.5', 'deprecation==2.1.0', 'aiobotocore==2.12.0', 'pylint==3.1.0', 'azure-cli-telemetry==1.0.8', 'astor==0.7.1', 'langsmith==0.1.19', 'flask-babel==3.0.1', 'ultralytics==8.1.16', 'fonttools==4.49.0', 'gradio==4.19.2', 'rpds-py==0.16.2', 'commonmark==0.8.1', 'zeep==4.2.1', 'poetry-plugin-export==1.6.0', 'telethon==1.34.0', 'azure-cli==2.57.0', 'ipywidgets==8.1.0', 'pbr==5.11.1', 'pyspark==3.5.0', 'mysql-connector-python==8.3.0', 'factory-boy==3.2.1', 'azure-mgmt-datamigration==10.0.0', 'pika==1.3.0', 'user-agent==0.1.8', 'parsedatetime==2.6', 'yarl==1.9.2', 'azure-mgmt-cognitiveservices==13.4.0', 'datetime==5.4', 'ipykernel==6.29.2', 'jpype1==1.5.0', 'websocket-client==1.6.4', 'pipx==1.4.3', 'ruamel-yaml==0.18.6', 'pyproject-api==1.5.4', 'googleapis-common-protos==1.62.0', 'charset-normalizer==3.3.1', 'lazy-object-proxy==1.10.0', 'stack-data==0.6.1', 'azure-mgmt-batch==17.1.0', 'mistune==3.0.1', 'azure-mgmt-servicefabric==1.0.0', 'apache-airflow-providers-imap==3.3.2', 'idna==3.6', 'cryptography==42.0.3', 'h5py==3.10.0', 'google-cloud-pubsub==2.19.4', 'ldap3==2.9', 'deepdiff==6.6.1', 'jsonschema==4.21.0', 'overrides==7.5.0', 'django-extensions==3.2.3', 'msrestazure==0.6.2', 'pyotp==2.9.0', 'oauthlib==3.2.1', 'hypothesis==6.98.11', 'spacy==3.7.4', 'dbt-extractor==0.5.1', 'exceptiongroup==1.2.0', 'flax==0.7.5', 'azure-mgmt-security==6.0.0', 'azure-mgmt-cosmosdb==9.3.0', 'nvidia-cufft-cu12==11.0.11.19', 'nvidia-cuda-runtime-cu12==12.3.101', 'requests-oauthlib==1.3.0', 'wandb==0.16.2', 'time-machine==2.13.0', 'thrift==0.15.0', 'ujson==5.8.0', 'pytest-xdist==3.4.0', 'keyring==24.3.1', 'sshtunnel==0.3.1', 'httpcore==1.0.3', 'opencensus==0.11.2', 'typed-ast==1.5.5', 'google-cloud-aiplatform==1.41.0', 'sphinxcontrib-applehelp==1.0.8', 'gcsfs==2024.2.0', 'commonmark==0.9.1', 'pycountry==23.12.7', 'azure-multiapi-storage==1.2.0', 'sentry-sdk==1.40.5', 'office365-rest-python-client==2.5.5', 'markdown-it-py==3.0.0', 'cycler==0.12.1', 'scikit-learn==1.4.0', 'office365-rest-python-client==2.5.6', 'ratelimit==2.1.0', 'pytorch-lightning==2.2.0', 'azure-kusto-ingest==4.3.0', 'tox==4.12.0', 'responses==0.24.1', 'text-unidecode==1.3', 'ec2-metadata==2.11.0', 'pluggy==1.3.0', 'colorama==0.4.4', 'geoip2==4.8.0', 'azure-mgmt-consumption==8.0.0', 'azure-mgmt-trafficmanager==1.0.0', 'pyrsistent==0.20.0', 'azure-mgmt-applicationinsights==3.0.0', 'emoji==2.10.0', 'black==24.2.0', 'pynacl==1.3.0', 'jmespath==0.10.0', 'orjson==3.9.15', 'uritemplate==4.1.1', 'pandas-gbq==0.21.0', 'azure-kusto-data==4.3.1', 'mypy-boto3-s3==1.34.14', 'nbclient==0.8.0', 'starkbank-ecdsa==2.0.3', 'azure-core==1.30.0', 'distro==1.7.0', 'smart-open==6.2.0', 'google-cloud-spanner==3.41.0', 'multidict==6.0.4', 'phonenumbers==8.13.31', 'gql==3.4.0', 'scipy==1.11.4', 'pooch==1.7.0', 'psutil==5.9.7', 'astor==0.8.1', 'notebook==7.1.0', 'charset-normalizer==3.3.2', 'django==4.2.8', 'asttokens==2.4.1', 'tensorboard==2.16.2', 'djangorestframework==3.13.0', 'pkginfo==1.10.0', 'streamlit==1.31.1', 'requests-oauthlib==1.3.1', 'nvidia-cusparse-cu12==12.1.2.141', 'google-cloud-bigtable==2.23.0', 'pillow==10.0.1', 'lightgbm==4.1.0', 'gevent==23.9.1', 'google-auth-oauthlib==1.2.0', 'bracex==2.3', 'execnet==2.0.1', 'agate==1.9.0', 'lxml==5.0.0', 'einops==0.6.1', 'requests-file==1.5.0', 'pytest-timeout==2.1.0', 'webencodings==0.4', 'pydeequ==1.2.0', 'pyzmq==25.1.1', 'pytest-mock==3.11.0', 'fuzzywuzzy==0.16.0', 'google-crc32c==1.2.0', 'azure-keyvault-keys==4.7.0', 'jsonlines==3.1.0', 'python-multipart==0.0.8', 'botocore-stubs==1.34.48', 'kiwisolver==1.4.5', 'azure-mgmt-policyinsights==1.0.0', 'requests-toolbelt==0.10.1', 'vine==1.3.0', 'click-plugins==1.0.4', 'types-protobuf==4.24.0.20240106', 'trio-websocket==0.10.4', 'awscli==1.32.55', 'mypy==1.8.0', 'widgetsnbextension==4.0.9', 'ujson==5.9.0', 'flask-jwt-extended==4.5.2', 'nbconvert==7.16.2', 'dnspython==2.5.0', 'google-cloud-audit-log==0.2.5', 'argon2-cffi==21.3.0', 'tensorflow-serving-api==2.13.1', 'pygments==2.17.0', 'nvidia-nvtx-cu12==12.3.101', 'dask==2024.1.1', 'azure-mgmt-datafactory==6.0.0', 'pytest-runner==6.0.1', 'locket==0.2.1', 'azure-eventgrid==4.16.0', 'service-identity==23.1.0', 'flask-jwt-extended==4.5.3', 'bleach==5.0.1', 'ipywidgets==8.1.1', 'opencensus-ext-azure==1.1.13', 'patsy==0.5.3', 'httplib2==0.21.0', 'shapely==2.0.3', 'xxhash==3.3.0', 'tomlkit==0.12.4', 'constructs==10.2.70', 'dateparser==1.1.8', 'qtconsole==5.5.1', 'google-cloud-core==2.3.3', 'flask-sqlalchemy==3.0.5', 'azure-synapse-accesscontrol==0.6.0', 'bandit==1.7.7', 'qtpy==2.4.1', 'pylint==3.0.3', 'azure-mgmt-netapp==10.1.0', 'shortuuid==1.0.11', 'pytest-xdist==3.3.1', 'contextlib2==0.5.5', 'cached-property==1.4.3', 'ml-dtypes==0.3.2', 'wtforms==3.1.0', 'numexpr==2.8.7', 'threadpoolctl==3.2.0', 'exceptiongroup==1.1.3', 'cloudpathlib==0.18.0', 'wtforms==3.1.1', 'django-filter==23.3', 'xlrd==2.0.1', 'mergedeep==1.3.3', 'dbt-core==1.7.7', 'langdetect==1.0.7', 'trio==0.24.0', 'fastapi==0.109.2', 'sphinxcontrib-htmlhelp==2.0.4', 'sniffio==1.3.0', 'freezegun==1.4.0', 'kornia==0.6.12', 'google-ads==22.1.0', 'kubernetes==28.1.0', 'webcolors==1.12', 'deprecated==1.2.12', 'widgetsnbextension==4.0.10', 'azure-mgmt-cdn==13.0.0', 'leather==0.3.4', 'configupdater==3.2', 'slackclient==2.9.2', 'retry==0.7.0', 'google-cloud-compute==1.16.1', 'botocore==1.34.48', 'safetensors==0.4.1', 'iso8601==2.1.0', 'annotated-types==0.4.0', 'types-awscrt==0.20.5', 'logbook==1.5.3', 'slack-sdk==3.27.0', 'requests-ntlm==1.1.0', 'send2trash==1.8.0', 'sphinx==7.2.5', 'fsspec==2024.2.0', 'tiktoken==0.6.0', 'progressbar2==4.4.0', 'openai==1.11.0', 'trio==0.23.1', 'networkx==3.2.1', 'azure-mgmt-datalake-analytics==0.6.0', 'azure-mgmt-cdn==12.0.0', 'gremlinpython==3.7.1', 'jupyter-core==5.7.1', 'tqdm==4.66.1', 'flask-appbuilder==4.3.11', 'azure-mgmt-apimanagement==3.0.0', 'fire==0.5.0', 'sphinxcontrib-devhelp==1.0.5', 'six==1.14.0', 'coloredlogs==15.0.1', 'jupyter-lsp==2.2.3', 'humanize==4.7.0', 'zope-event==5.0', 'shap==0.44.1', 'db-dtypes==1.2.0', 'click-repl==0.3.0', 'leather==0.3.3', 'requests==2.29.0', 'poetry-plugin-export==1.5.0', 'google-cloud-datacatalog==3.18.0', 'google-cloud-dataproc==5.9.1', 'funcsigs==1.0.1', 'knack==0.10.1', 'types-setuptools==69.1.0.20240223', 'psycopg==3.1.18', 'azure-multiapi-storage==1.0.0', 'rapidfuzz==3.6.0', 'patsy==0.5.6', 'azure-mgmt-advisor==3.0.0', 'sphinxcontrib-applehelp==1.0.7', 'configupdater==3.1', 'azure-mgmt-synapse==1.0.0', 'azure-mgmt-botservice==2.0.0', 'protobuf3-to-dict==0.1.3', 'pysocks==1.6.8', 'tensorflow-metadata==1.14.0', 'types-requests==2.31.0.20240125', 'grpcio==1.60.1', 'evergreen-py==3.6.22', 'azure-mgmt-botservice==0.3.0', 'aiohttp==3.9.1', 'sagemaker==2.208.0', 'opentelemetry-exporter-otlp-proto-grpc==1.21.0', 'kr8s==0.13.4', 'bytecode==0.15.0', 'libclang==15.0.6.1', 'scandir==1.9.0', 'pathos==0.3.1', 'jsonpickle==3.0.1', 'ratelimit==2.2.0', 'semver==3.0.2', 'types-setuptools==69.1.0.20240301', 'jupyterlab-widgets==3.0.10', 'onnxruntime==1.17.0', 'jsonpatch==1.32', 'apache-airflow-providers-databricks==6.1.0', 'pathspec==0.12.0', 'prometheus-client==0.20.0', 'execnet==2.0.2', 'sentencepiece==0.1.98', 'tiktoken==0.5.2', 'azure-storage-file-share==12.15.0', 'tornado==6.4', 'filelock==3.13.1', 'soupsieve==2.4.1', 'humanfriendly==9.1', 'wrapt==1.16.0', 'ndg-httpsclient==0.5.0', 'dvclive==3.42.0', 'marshmallow-dataclass==8.5.13', 'pandas==2.2.1', 'trio==0.23.2', 'snowflake-sqlalchemy==1.4.7', 'requests-mock==1.10.0', 'cfn-lint==0.85.1', 'google-cloud-container==2.41.0', 'azure-mgmt-batchai==1.0.1', 'trove-classifiers==2024.2.23', 'azure-mgmt-authorization==3.0.0', 'thinc==8.2.3', 'databricks-api==0.8.0', 'pypdf==4.0.0', 'jsonlines==3.0.0', 'secretstorage==3.3.1', 'asyncio==3.4.2', 'addict==2.4.0', 'aiofiles==23.2.1', 'marshmallow-enum==1.5.1', 'nose==1.3.4', 'grpcio-status==1.60.0', 'spacy-legacy==3.0.12', 'aioitertools==0.9.0', 'prometheus-client==0.19.0', 'tblib==1.7.0', 'db-contrib-tool==0.6.12', 'tensorflow-metadata==1.13.0', 'urllib3==2.2.1', 'pgpy==0.5.3', 'tensorflow-text==2.15.0', 'dataclasses-json==0.6.4', 'dill==0.3.8', 'azure-keyvault-secrets==4.6.0', 'dill==0.3.6', 'sphinxcontrib-serializinghtml==1.1.8', 'google-cloud-container==2.40.0', 'jsondiff==1.3.0', 'ply==3.10', 'annotated-types==0.5.0', 'hyperframe==6.0.0', 'identify==2.5.35', 'httplib2==0.20.4', 'pydantic-core==2.16.2', 'sphinxcontrib-qthelp==1.0.7', 'snowflake-connector-python==3.7.0', 'azure-cli==2.55.0', 'enum34==1.1.8', 'pyyaml==5.4.1', 'ipywidgets==8.1.2', 'json5==0.9.18', 'reportlab==4.0.8', 'filelock==3.12.4', 'portalocker==2.7.0', 'jedi==0.18.2', 'cryptography==42.0.4', 'lxml==5.0.1', 'webcolors==1.11.1', 'jupyter-server==2.12.3', 'simple-salesforce==1.12.3', 'mock==5.0.1', 'scikit-learn==1.3.1', 'telethon==1.33.0', 'websocket-client==1.6.3', 'ptyprocess==0.5.2', 'pymongo==4.6.2', 'datadog-api-client==2.22.0', 'google-cloud-storage==2.14.0', 'uamqp==1.6.8', 'cog==0.9.4', 'python-utils==3.8.1', 'langchain==0.1.10', 'py==1.9.0', 'uvicorn==0.26.0', 'jsonpointer==2.3', 'diskcache==5.6.3', 'oauthlib==3.2.2', 'smmap==5.0.1', 'dvc-render==0.7.0', 'configobj==5.0.8', 'python-jose==3.2.0', 'db-contrib-tool==0.6.11', 'importlib-metadata==7.0.1', 'mlflow==2.10.1', 'mmh3==4.1.0', 'tzlocal==5.1', 'rsa==4.7.2', 'cinemagoer==2023.5.1', 'flask-session==0.4.1', 'azure-data-tables==12.5.0', 'azure-mgmt-trafficmanager==0.51.0', 'msgpack==1.0.7', 'jupyterlab-pygments==0.3.0', 'threadpoolctl==3.1.0', 'opentelemetry-sdk==1.21.0', 'pre-commit==3.6.2', 'pipx==1.4.1', 'azure-mgmt-monitor==6.0.2', 'azure-keyvault-keys==4.8.0', 'pytest-rerunfailures==13.0', 'python-dateutil==2.8.2', 'pytest==8.0.1', 'paramiko==3.4.0', 'constructs==10.3.0', 'databricks-api==0.9.0', 'tensorflow-text==2.13.0', 'scipy==1.11.3', 'schema==0.7.4', 'apache-airflow-providers-imap==3.5.0', 'hypothesis==6.98.17', 'azure-mgmt-imagebuilder==1.3.0', 'applicationinsights==0.11.8', 'oscrypto==1.2.1', 'simplejson==3.19.2', 'frozenlist==1.3.3', 'cx-oracle==8.2.1', 'pyaml==23.9.6', 'google-pasta==0.1.7', 'ecdsa==0.16.1', 'boltons==23.0.0', 'jmespath==1.0.0', 'pyhcl==0.4.3', 'fastparquet==2023.10.0', 'ruff==0.3.0', 'xmltodict==0.13.0', 'simplejson==3.18.4', 'torch==2.1.2', 'kombu==5.3.3', 'pytest-html==4.0.2', 'tzdata==2023.4', 'geopy==2.3.0', 'ijson==3.2.3', 'jmespath==1.0.1', 'babel==2.13.0', 'google-cloud-core==2.4.1', 'parso==0.8.3', 'bcrypt==4.1.1', 'coverage==7.4.3', 'isort==5.13.2', 'babel==2.14.0', 'incremental==21.3.0', 'h2==4.0.0', 'entrypoints==0.4', 'jaxlib==0.4.25', 'nose==1.3.7', 'distlib==0.3.7', 'bracex==2.4', 'python-docx==1.1.0', 'mypy-boto3-redshift-data==1.33.0', 'mpmath==1.2.1', 'protobuf==4.25.3', 'moto==5.0.1', 'catalogue==2.0.8', 'azure-cli-telemetry==1.1.0', 'python-levenshtein==0.25.0', 'jedi==0.19.0', 'dnspython==2.6.1', 'gremlinpython==3.6.6', 'korean-lunar-calendar==0.2.0', 'pyasn1==0.5.0', 'keras-applications==1.0.8', 'msal-extensions==1.0.0', 'azure-mgmt-containerregistry==10.2.0', 'wheel==0.41.2', 'huggingface-hub==0.21.3', 'azure-mgmt-datalake-analytics==0.4.0', 'ddtrace==2.6.5', 'tensorflow-datasets==4.9.1', 'apache-airflow-providers-snowflake==5.3.1', 'python-slugify==8.0.3', 'mypy-boto3-rds==1.34.50', 'semantic-version==2.10.0', 'python-dotenv==0.21.1', 'editables==0.3', 'pytest-forked==1.6.0', 'nvidia-nccl-cu12==2.19.3', 'apache-airflow-providers-ssh==3.9.0', 'azure-eventhub==5.11.6', 'sphinxcontrib-htmlhelp==2.0.3', 'gast==0.5.2', 'python-box==7.1.0', 'cmake==3.27.9', 'scandir==1.8', 'oscrypto==1.3.0', 'azure-data-tables==12.4.3', 'selenium==4.17.2', 'mccabe==0.6.0', 'setuptools-rust==1.8.1', 'pox==0.3.2', 'apache-airflow-providers-sqlite==3.7.0', 'bleach==6.0.0', 'pooch==1.8.1', 'installer==0.5.1', 'flask-login==0.6.3', 'poetry==1.7.0', 'dulwich==0.21.7', 'opentelemetry-exporter-otlp==1.23.0', 'huggingface-hub==0.21.1', 'docopt==0.6.2', 'cfgv==3.3.1', 'streamlit==1.31.0', 'invoke==2.2.0', 'azure-mgmt-loganalytics==10.0.0', 'pytimeparse==1.1.8', 'pymeeus==0.5.12', 'pyparsing==3.1.1', 'emoji==2.9.0', 'sqlalchemy-utils==0.41.1', 'apache-airflow-providers-http==4.9.1', 'watchfiles==0.19.0', 'azure-multiapi-storage==1.1.0', 'jupyter-console==6.6.1', 'django==4.2.11', 'jaraco-classes==3.3.1', 'async-lru==2.0.3', 'google-cloud-core==2.4.0', 'torchmetrics==1.3.1', 'python-editor==1.0.1', 'tenacity==8.2.1', 'py-cpuinfo==9.0.0', 'azure-mgmt-hdinsight==7.0.0', 'ptyprocess==0.6.0', 'ruamel-yaml-clib==0.2.8', 'contextlib2==21.6.0', 'botocore-stubs==1.34.53', 'tifffile==2024.1.30', 'google-cloud-secret-manager==2.18.2', 'lazy-object-proxy==1.9.0', 'google==3.0.0', 'apache-airflow-providers-http==4.8.0', 'tensorboard-data-server==0.7.1', 'elastic-transport==8.12.0', 'dacite==1.8.0', 'ultralytics==8.1.20', 'lz4==4.3.2', 'dulwich==0.21.5', 'pyodbc==5.0.0', 'monotonic==1.6', 'tensorflow-serving-api==2.14.0', 'azure-mgmt-eventgrid==10.2.0', 'twilio==9.0.0', 'azure-keyvault==4.2.0', 'azure-mgmt-datafactory==5.0.0', 'azure-storage-file-share==12.14.2', 'scramp==1.4.3', 'flask==3.0.0', 'nbclient==0.9.0', 'mypy-boto3-appflow==1.29.0', 'google-cloud-bigquery-storage==2.22.0', 'flake8==6.0.0', 'gcsfs==2023.12.0', 'sentry-sdk==1.40.4', 'python-daemon==3.0.1', 'pep517==0.13.0', 'azure-cli-core==2.58.0', 'pytest==8.0.2', 'cloudpickle==2.2.1', 'great-expectations==0.18.7', 'nvidia-cusparse-cu12==12.2.0.103', 'python-utils==3.8.0', 'bcrypt==4.1.2', 'uri-template==1.2.0', 'texttable==1.6.6', 'inflect==6.2.0', 'bleach==6.1.0', 'pytest-forked==1.4.0', 'connexion==3.0.5', 'azure-mgmt-search==9.0.0', 'pytimeparse==1.1.7', 'smart-open==7.0.0', 'configparser==5.3.0', 'mistune==3.0.2', 'awscli==1.32.54', 'cmdstanpy==1.1.0', 'redis==5.0.2', 'nbconvert==7.16.1', 'datadog==0.46.0', 'atomicwrites==1.4.0', 'pydash==7.0.7', 'torch==2.2.0', 'djangorestframework==3.13.1', 'pytest-asyncio==0.23.3', 'scikit-image==0.22.0', 'grpcio-health-checking==1.62.0', 'opencv-python-headless==4.8.1.78', 'mypy-extensions==1.0.0', 'pydub==0.24.1', 'shap==0.44.0', 'azure-mgmt-dns==3.0.0', 'apache-airflow-providers-slack==8.6.0', 'msrestazure==0.6.4', 'traitlets==5.13.0', 'flask-sqlalchemy==3.1.0', 'azure-core==1.29.6', 'sentence-transformers==2.5.0', 'adal==1.2.7', 'pycares==4.2.2', 'onnx==1.15.0', 'pyasn1-modules==0.2.8', 'uvicorn==0.27.0', 'urllib3==2.1.0', 'unittest-xml-reporting==3.2.0', 'bitarray==2.9.2', 'click==8.1.7', 'awswrangler==3.5.2', 'ua-parser==0.18.0', 'yarl==1.9.3', 'factory-boy==3.3.0', 'pyproject-api==1.6.1', 'pyflakes==3.2.0', 'tifffile==2024.2.12', 'pathspec==0.11.2', 'azure-mgmt-datamigration==9.0.0', 'monotonic==1.5', 'scikit-image==0.20.0', 'google-auth-oauthlib==1.0.0', 'azure-mgmt-appconfiguration==3.0.0', 'prison==0.1.3', 'google-auth-oauthlib==1.1.0', 'tensorflow-io-gcs-filesystem==0.35.0', 'elasticsearch-dsl==8.12.0', 'msal-extensions==0.3.1', 'flask-babel==4.0.0', 'jinja2==3.1.2', 'ray==2.9.2', 'gensim==4.3.2', 'greenlet==3.0.2', 'keras-preprocessing==1.1.1', 'wheel==0.42.0', 'diskcache==5.6.0', 'mako==1.2.4', 'jsonschema==4.21.1', 'ply==3.9', 'pymysql==1.1.0', 'flask-caching==2.0.2', 'distributed==2024.2.0', 'azure-mgmt-eventgrid==10.1.0', 'limits==3.8.0', 'pyzmq==25.1.0', 'python-box==7.0.1', 'poetry==1.8.2', 'pyserial==3.5', 'qtpy==2.4.0', 'javaproperties==0.7.0', 'async-generator==1.8', 'rfc3339-validator==0.1.4', 'jupyter-server==2.12.4', 'ddsketch==2.0.2', 'transformers==4.38.1', 'netaddr==1.1.0', 'botocore==1.34.47', 'authlib==1.2.1', 'azure-mgmt-sqlvirtualmachine==0.3.0', 'polars==0.20.13', 'terminado==0.18.0', 'paramiko==3.3.1', 'azure-storage-common==1.4.2', 'python-slugify==8.0.4', 'azure-keyvault-secrets==4.8.0', 'pyspark==3.5.1', 'docker-pycreds==0.4.0', 'accelerate==0.27.1', 'kornia==0.7.0', 'aws-psycopg2==1.3.8', 'decorator==5.1.1', 'bracex==2.2.1', 'jupyter-core==5.7.0', 'frozendict==2.3.9', 'prompt-toolkit==3.0.41', 'asn1crypto==1.5.0', 'tzlocal==5.2', 'python-levenshtein==0.23.0', 'types-setuptools==69.1.0.20240215', 'google-ads==23.1.0', 'flask-session==0.5.0', 'marshmallow==3.21.1', 'checkov==3.2.25', 'pyrsistent==0.19.3', 'fastavro==1.9.4', 'aniso8601==9.0.0', 'flask-wtf==1.1.2', 'hologram==0.0.15', 'xxhash==3.4.1', 'azure-mgmt-sqlvirtualmachine==0.5.0', 'pgpy==0.5.4', 'azure-mgmt-loganalytics==12.0.0', 'webencodings==0.5', 'pyproj==3.6.1', 'cssselect==1.1.0', 'sniffio==1.3.1', 'rapidfuzz==3.6.1', 'cachelib==0.11.0', 'pbr==6.0.0', 'einops==0.7.0', 'google-api-core==2.17.1', 'awswrangler==3.6.0', 'cramjam==2.7.0', 'google-crc32c==1.3.0', 'beautifulsoup4==4.12.1', 'importlib-resources==6.1.2', 'boto3==1.34.54', 'proto-plus==1.22.2', 'types-python-dateutil==2.8.19.20240106', 'boto3==1.34.53', 'lockfile==0.10.2', 'executing==1.1.1', 'mpmath==1.3.0', 'azure-mgmt-signalr==1.1.0', 'bitarray==2.9.0', 'py-cpuinfo==7.0.0', 'google-cloud-logging==3.7.0', 'partd==1.3.0', 'sympy==1.11.1', 'applicationinsights==0.11.10', 'linkify-it-py==2.0.3', 'nvidia-curand-cu12==10.3.4.52', 'requests-mock==1.11.0', 'polars==0.20.10', 'debugpy==1.8.1', 'packaging==23.1', 'mypy-boto3-rds==1.34.49', 'sagemaker==2.209.0', 'typeguard==4.1.4', 'reportlab==4.1.0', 'spark-nlp==5.2.1', 'pendulum==2.1.2', 'cdk-nag==2.28.44', 'plotly==5.18.0', 'apscheduler==3.10.3', 'celery==5.3.6', 'junit-xml==1.9', 'pyperclip==1.8.1', 'tableauserverclient==0.30', 'mypy-boto3-rds==1.34.44', 'levenshtein==0.24.0', 'hatchling==1.21.0', 'terminado==0.17.1', 'opencensus==0.11.3', 'azure-mgmt-authorization==2.0.0', 'typing==3.7.4.3', 'pymsteams==0.2.1', 'opentelemetry-api==1.23.0', 'numexpr==2.8.6', 'requests-oauthlib==1.2.0', 'jupyter-server-terminals==0.5.1', 'cfn-lint==0.85.3', 'appdirs==1.4.4', 'fabric==3.2.2', 'requests-ntlm==1.0.0', 'azure-storage-blob==12.18.2', 'tensorflow-estimator==2.15.0', 'typing==3.7.4', 'azure-mgmt-dns==8.0.0', 'pyflakes==3.0.1', 'flatbuffers==23.5.26', 'opensearch-py==2.4.1', 'antlr4-python3-runtime==4.13.0', 'aniso8601==8.1.1', 'jupyter-events==0.8.0', 'rdflib==7.0.0', 'boto3-stubs==1.34.47', 'progressbar2==4.3.2', 'unidecode==1.3.7', 'poetry-core==1.8.0', 'pypdf==4.1.0', 'python-magic==0.4.25', 'send2trash==1.8.2', 'imageio==2.33.1', 'asgiref==3.6.0', 'twisted==24.3.0', 'sqlparse==0.4.3', 'anyio==4.1.0', 'python-gitlab==4.3.0', 'dulwich==0.21.6', 'deepdiff==6.7.1', 'invoke==2.1.3', 'cfn-lint==0.85.2', 'qtconsole==5.4.4', 'django-filter==23.4', 'parsedatetime==2.4', 'traitlets==5.14.1', 'qtpy==2.3.1', 'grpcio-status==1.60.1', 'diskcache==5.6.1', 'pydub==0.25.1', 'pydantic-core==2.16.1', 'ldap3==2.8.1', 'requests==2.31.0', 'azure-identity==1.15.0', 'ftfy==6.1.1', 'google-crc32c==1.5.0', 'azure-mgmt-cdn==11.0.0', 'validators==0.21.1', 'azure-mgmt-marketplaceordering==1.0.0', 'kfp==2.6.0', 'websockets==11.0.2', 'plotly==5.19.0', 'blis==0.9.1', 'pytest-cov==4.0.0', 'build==1.0.0', 'redshift-connector==2.1.0', 'ninja==1.11.1', 'asynctest==0.13.0', 'pillow==10.2.0', 'portalocker==2.8.2', 'google-cloud-bigquery-storage==2.23.0', 'pipenv==2023.12.1', 'azure-storage-queue==12.7.3', 'ninja==1.11.1.1', 'cryptography==42.0.5', 'pyspnego==0.10.2', 'pathlib2==2.3.6', 'grpcio-health-checking==1.60.1', 'types-awscrt==0.20.4', 'blinker==1.6.2', 'azure-mgmt-loganalytics==11.0.0', 'cachetools==5.3.0', 'cssselect==1.0.3', 'azure-mgmt-advisor==4.0.0', 'opentelemetry-sdk==1.22.0', 'gspread==6.0.2', 'pycparser==2.19', 'markupsafe==2.1.5', 'huggingface-hub==0.20.2', 'uamqp==1.6.7', 'azure-mgmt-redhatopenshift==1.3.0', 'holidays==0.43', 'langsmith==0.1.7', 'platformdirs==4.0.0', 'datetime==5.2', 'parso==0.8.1', 'service-identity==21.1.0', 'python-jose==3.3.0', 'pyzmq==25.1.2', 'great-expectations==0.18.8', 'sentry-sdk==1.40.6', 'azure-mgmt-apimanagement==4.0.0', 'authlib==1.2.0', 'confluent-kafka==2.1.1', 'portalocker==2.8.1', 'cerberus==1.3.4', 'azure-mgmt-web==7.0.0', 'azure-mgmt-cognitiveservices==13.5.0', 'convertdate==2.4.0', 'jira==3.5.2', 'google-cloud-compute==1.16.0', 'cached-property==1.5.2', 'ipython==8.17.2', 'pytest-html==4.1.0', 'stringcase==1.2.0', 'xlsxwriter==3.2.0', 'statsmodels==0.14.1', 'pybind11==2.11.0', 'jupyter-lsp==2.2.1', 'pymysql==1.0.2', 'azure-servicebus==7.11.2', 'cachecontrol==0.13.1', 'azure-mgmt-eventhub==10.0.0', 'twisted==22.10.0', 'apispec==6.5.0', 'numba==0.58.1', 'oauth2client==4.1.3', 'hiredis==2.2.3', 'pyarrow==15.0.0', 'ultralytics==8.1.22', 'schema==0.7.5', 'pytest-xdist==3.5.0', 'keyring==24.3.0', 'aioitertools==0.10.0', 'smmap==4.0.0', 'cmake==3.28.3', 'ninja==1.10.2.4', 'ipykernel==6.29.1', 'azure-storage-file-datalake==12.14.0', 'envier==0.5.1', 'networkx==3.2', 'spark-nlp==5.2.2', 'rich==13.5.3', 'aws-lambda-powertools==2.34.2', 'azure-mgmt-synapse==0.8.0', 'azure-mgmt-trafficmanager==1.1.0', 'azure-mgmt-maps==1.0.0', 'azure-mgmt-iothub==3.0.0', 'redis==5.0.1', 'twisted==23.8.0', 'async-lru==2.0.4', 'faker==23.2.1', 'json5==0.9.20', 'opentelemetry-proto==1.21.0', 'azure-cli-core==2.56.0', 'awscli==1.32.48', 'nvidia-cudnn-cu12==8.9.5.30', 'apache-airflow-providers-sqlite==3.7.1', 'pypdf==4.0.2', 's3fs==2024.2.0', 'configobj==5.0.5', 'jinja2==3.1.3', 'chex==0.1.85', 'aws-xray-sdk==2.12.0', 'zope-interface==6.0', 'antlr4-python3-runtime==4.13.1', 'sqlalchemy-jsonfield==0.9.0', 'atomicwrites==1.4.1', 'uc-micro-py==1.0.2', 'python-http-client==3.3.7', 'pycryptodome==3.19.1', 'tifffile==2023.12.9', 'pymongo==4.6.0', 'pkginfo==1.9.5', 'azure-mgmt-datalake-store==0.3.0', 'toml==0.10.0', 'hpack==4.0.0', 'azure-mgmt-kusto==3.1.0', 'prometheus-client==0.18.0', 'google-auth==2.28.1', 'pyperclip==1.8.2', 'protobuf==4.25.2', 'fasteners==0.19', 'dvclive==3.44.0', 'mock==5.1.0', 'thrift==0.14.2', 'murmurhash==1.0.10', 'pytest-timeout==2.0.2', 'pathos==0.3.0', 'azure-eventhub==5.11.5', 'opensearch-py==2.4.0', 'limits==3.7.0', 'apache-airflow-providers-amazon==8.18.0', 'mypy==1.7.0', 'cligj==0.7.1', 'uritemplate==3.0.1', 'prompt-toolkit==3.0.42', 'graphql-core==3.2.2', 'opentelemetry-exporter-otlp-proto-grpc==1.22.0', 'pytest-mock==3.11.1', 'tensorboard==2.16.1', 'databricks-sql-connector==3.1.0', 'marshmallow-sqlalchemy==1.0.0', 'jupyterlab-pygments==0.2.1', 'datasets==2.17.0', 'pywavelets==1.5.0', 'pysocks==1.7.0', 'snowballstemmer==2.0.0', 'opentelemetry-exporter-otlp-proto-common==1.21.0', 'marshmallow-dataclass==8.5.14', 'pexpect==4.7.0', 'unittest-xml-reporting==3.0.4', 'xmltodict==0.12.0', 'antlr4-python3-runtime==4.12.0', 'ddsketch==2.0.4', 'google-api-python-client==2.120.0', 'apache-airflow==2.8.1', 'stringcase==1.0.6', 'smart-open==7.0.1', 'sphinxcontrib-serializinghtml==1.1.10', 'docker==7.0.0', 'cython==3.0.8', 'unicodecsv==0.14.1', 'cachetools==5.3.2', 'pathy==0.11.0', 'typeguard==4.1.3', 'pickleshare==0.7.4', 'pip-tools==7.4.0', 'azure-mgmt-iotcentral==4.1.0', 'nvidia-cudnn-cu12==8.9.6.50', 'pypdf==4.0.1', 'regex==2023.10.3', 'kfp==2.5.0', 'datadog==0.47.0', 'elasticsearch==8.11.1', 'azure-mgmt-datafactory==4.0.0', 'prison==0.2.1', 'azure-batch==14.0.0', 'msgpack==1.0.5', 'python-magic==0.4.26', 'colorlog==6.8.2', 'azure-synapse-spark==0.7.0', 'click-repl==0.1.6', 'cleo==2.1.0', 'notebook==7.1.1', 'xyzservices==2023.10.1', 'apache-airflow-providers-common-sql==1.10.1', 'pg8000==1.30.3', 'docker==6.1.3', 'sqlalchemy==2.0.26', 'adal==1.2.6', 'sphinx-rtd-theme==1.2.2', 'cinemagoer==2022.12.27', 'accelerate==0.27.0', 'installer==0.7.0', 'parameterized==0.8.1', 'langsmith==0.1.6', 'google-cloud-bigquery==3.18.0', 'pyelftools==0.29', 'comm==0.2.0', 'apache-airflow-providers-slack==8.6.1', 'ultralytics==8.1.23', 'murmurhash==1.0.8', 'rich-argparse==1.3.0', 'time-machine==2.14.0', 'pexpect==4.9.0', 'mysql-connector-python==8.1.0', 'referencing==0.32.1', 'xarray==2024.1.0', 'pyodbc==5.0.1', 'setuptools-rust==1.9.0', 'google-auth==2.28.0', 'parse==1.20.1', 'checkov==3.2.30', 'asgiref==3.7.1', 'secretstorage==3.3.2', 'spacy==3.7.1', 'pytest-metadata==3.0.0', 'ultralytics==8.1.17', 'nodeenv==1.6.0', 'aiosignal==1.1.2', 'pyasn1==0.5.1', 'pydata-google-auth==1.8.2', 'apache-airflow==2.8.0', 'pyee==11.1.0', 'kfp-server-api==2.0.5', 'azure-mgmt-devtestlabs==4.0.0', 'feedparser==6.0.10', 'snowflake-connector-python==3.6.0', 'google-cloud-firestore==2.15.0', 'responses==0.25.0', 'jupyter-lsp==2.2.0', 'sqlalchemy-jsonfield==1.0.2', 'aws-sam-translator==1.83.0', 'pydata-google-auth==1.8.0', 'azure-mgmt-kusto==3.2.0', 'dacite==1.7.0', 'sqlalchemy-utils==0.41.0', 'prompt-toolkit==3.0.43', 'sentence-transformers==2.3.0', 'nodeenv==1.8.0', 'ndg-httpsclient==0.5.1', 'nvidia-nvjitlink-cu12==12.3.101', 'yamllint==1.35.1', 'snowflake-sqlalchemy==1.5.0', 'build==1.0.3', 'logbook==1.7.0', 'snowflake-sqlalchemy==1.5.1', 'astor==0.8.0', 'pytest-timeout==2.2.0', 'nbconvert==7.15.0', 'h11==0.13.0', 'click-didyoumean==0.3.0', 'pycodestyle==2.10.0', 'dataclasses==0.4', 'yarl==1.9.4', 'pysftp==0.2.8', 'azure-mgmt-storage==20.1.0', 'cfgv==3.3.0', 'cachelib==0.12.0', 'azure-mgmt-security==5.0.0', 'azure-mgmt-batch==17.0.0', 'azure-mgmt-servicefabric==2.0.0', 'prometheus-flask-exporter==0.23.0', 'fqdn==1.5.0', 'google-cloud-pubsub==2.19.6', 'pandocfilters==1.5.1', 'tabulate==0.8.10', 'newrelic==9.7.0', 'pyotp==2.8.0', 'deprecated==1.2.14', 'ansible-core==2.15.7', 'docker==6.1.2', 'accelerate==0.27.2', 'outcome==1.1.0', 'confluent-kafka==2.3.0', 'azure-mgmt-cosmosdb==9.4.0', 'distributed==2024.2.1', 'argon2-cffi==21.2.0', 'chex==0.1.84', 'pydantic==2.6.2', 'msrest==0.6.19', 'pg8000==1.30.4', 'ptyprocess==0.7.0', 'azure-keyvault-administration==4.2.0', 'google-cloud-logging==3.8.0', 'html5lib==1.1', 'maxminddb==2.5.1', 'natsort==8.4.0', 'flatbuffers==23.5.9', 'hypothesis==6.98.12', 'google-auth==2.27.0', 'termcolor==2.4.0', 'google-cloud-datacatalog==3.18.1', 'joblib==1.3.2', 'python-docx==1.0.1', 'boto==2.47.0', 'referencing==0.33.0', 'nvidia-cuda-cupti-cu12==12.2.142', 'rfc3986==1.5.0', 'nvidia-cuda-cupti-cu12==12.3.52', 'lightning-utilities==0.9.0', 'xlwt==1.3.0', 'scikit-learn==1.3.2', 'sentry-sdk==1.40.3', 'python-levenshtein==0.24.0', 'tornado==6.3.3', 'locket==0.2.0', 'sqlparse==0.4.4', 'ipykernel==6.29.3', 'preshed==3.0.8', 'javaproperties==0.8.1', 'typing-extensions==4.8.0', 'google-re2==0.2.20220601', 'agate==1.9.1', 'pyee==11.0.0', 'semantic-version==2.9.0', 'nodeenv==1.7.0', 'regex==2023.8.8', 'httpcore==1.0.4', 'pyparsing==3.0.9', 'nvidia-cusolver-cu12==11.5.2.141', 'catalogue==2.0.9', 'sqlalchemy-utils==0.40.0', 'ipython==8.18.1', 'aws-requests-auth==0.4.2', 'rfc3339-validator==0.1.3', 'jsonpath-ng==1.6.0', 'tableauserverclient==0.29', 'ordered-set==4.0.1', 'gensim==4.3.1', 'google-cloud-secret-manager==2.18.1', 'robotframework-seleniumlibrary==6.1.3', 'inflection==0.5.1', 'sentencepiece==0.1.99', 'pyarrow==14.0.1', 'redshift-connector==2.0.918', 'anyio==4.3.0', 'dbt-core==1.7.6', 'azure-synapse-spark==0.5.0', 'werkzeug==3.0.0', 'tomli==1.2.3', 's3fs==2023.12.2', 'tabulate==0.9.0', 'graphql-core==3.2.3', 'werkzeug==3.0.1', 'xyzservices==2023.10.0', 'nbclassic==0.5.5', 'wcwidth==0.2.13', 'click-plugins==1.1', 'google-api-python-client==2.119.0', 'aioitertools==0.11.0', 'pywavelets==1.2.0', 'hypothesis==6.98.16', 'evaluate==0.3.0', 'spark-nlp==5.2.3', 'setproctitle==1.3.1', 'semver==3.0.0', 'hdfs==2.7.2', 'asn1crypto==1.4.0', 'preshed==3.0.7', 'matplotlib-inline==0.1.3', 'azure-mgmt-compute==30.4.0', 'pyserial==3.4', 'mpmath==1.1.0', 'contextlib2==0.6.0', 'nest-asyncio==1.5.9', 'ml-dtypes==0.3.0', 'arrow==1.3.0', 'jsonpointer==2.2', 'types-pytz==2024.1.0.20240203', 'idna==3.5', 'pypandoc==1.12', 'pyee==11.0.1', 'fastjsonschema==2.18.1', 'msal==1.25.0', 'sphinxcontrib-applehelp==1.0.6', 's3transfer==0.9.0', 'langchain==0.1.8', 'pytest-runner==5.3.2', 'mashumaro==3.11', 'sortedcontainers==2.3.0', 'srsly==2.4.8', 'apache-airflow==2.7.3', 'azure-storage-queue==12.9.0', 'nvidia-cuda-runtime-cu12==12.2.140', 'spacy-loggers==1.0.5', 'retrying==1.3.3', 'opentelemetry-exporter-otlp-proto-common==1.23.0', 'azure-mgmt-containerinstance==10.0.0', 'msal==1.27.0', 'mock==5.0.2', 'elastic-transport==8.11.0', 'tomlkit==0.12.3', 'idna==3.4', 'pooch==1.8.0', 'xgboost==2.0.1', 'jupyterlab==4.1.0', 'poetry==1.8.1', 'pyproject-api==1.6.0', 'uvloop==0.18.0', 'ipykernel==6.29.0', 'trio-websocket==0.10.3', 'setproctitle==1.3.3', 'sphinx-rtd-theme==2.0.0', 'pymssql==2.2.9', 'pydeequ==1.1.0', 'tensorflow-estimator==2.14.0', 'bokeh==3.3.4', 'imbalanced-learn==0.12.0', 'tensorflow-metadata==1.13.1', 'ratelimit==2.2.1', 'aniso8601==9.0.1', 'azure-mgmt-recoveryservicesbackup==7.0.0', 'ordered-set==4.0.2', 'faker==23.1.0', 'gradio-client==0.10.1', 'tomlkit==0.12.1', 'azure-mgmt-kusto==3.3.0', 'flask-limiter==3.5.1', 'azure-mgmt-batchai==2.0.0', 'sh==2.0.6', 'nltk==3.8', 'azure-data-tables==12.4.4', 'apispec==6.4.0', 'nbclassic==1.0.0', 'pytest==8.0.0', 'pygments==2.17.2', 'ruamel-yaml==0.18.3', 'pytest-metadata==3.1.0', 'azure-graphrbac==0.61.1', 'shellingham==1.5.3', 'wheel==0.41.3', 'smmap==5.0.0', 'apache-airflow-providers-databricks==6.0.0', 'docker-pycreds==0.2.3', 'starlette==0.36.3', 'orjson==3.9.13', 'python-json-logger==2.0.5', 'pure-eval==0.2.1', 'overrides==7.6.0', 'python-slugify==8.0.2', 'proto-plus==1.23.0', 'tqdm==4.66.0', 'boto3-stubs==1.34.48', 'readme-renderer==41.0', 'psutil==5.9.8', 'azure-mgmt-maps==2.0.0', 'azure-mgmt-iothub==2.4.0', 'pathlib==0.97', 'azure-keyvault-keys==4.9.0', 'jsondiff==1.3.1', 'huggingface-hub==0.21.2', 'spacy-legacy==3.0.11', 'nbconvert==7.16.0', 'cycler==0.12.0', 'agate==1.8.0', 'cdk-nag==2.28.54', 'azure-mgmt-monitor==6.0.1', 'mashumaro==3.10', 'gunicorn==21.0.1', 'makefun==1.15.1', 'blinker==1.6.3', 'azure-synapse-accesscontrol==0.7.0', 'xlrd==2.0.0', 'qtconsole==5.5.0', 'hypothesis==6.98.10', 'watchtower==3.0.1', 'db-dtypes==1.1.0', 'appdirs==1.4.3', 'asttokens==2.3.0', 'traitlets==5.14.0', 'async-generator==1.10', 'timm==0.9.16', 'pymysql==1.0.3', 'ddtrace==2.7.0', 'sendgrid==6.9.7', 'kafka-python==2.0.0', 'seaborn==0.13.0', 'xgboost==2.0.3', 'backoff==2.1.2', 'pillow==10.1.0', 'avro==1.11.3', 'parameterized==0.9.0', 'bandit==1.7.5', 'ansible==8.6.0', 'evidently==0.4.15', 'uri-template==1.1.0', 'spacy-loggers==1.0.4', 'azure-mgmt-rdbms==10.0.0', 'webcolors==1.13', 'ruff==0.2.1', 'attrs==22.2.0', 'tensorstore==0.1.53', 'mysql-connector-python==8.2.0', 'huggingface-hub==0.20.3', 'lightning-utilities==0.10.1', 'cffi==1.15.0', 'dbt-core==1.7.9', 'google-cloud-dlp==3.15.0', 'autopep8==2.0.4', 'pox==0.3.3', 'azure-storage-blob==12.18.3', 'python-dateutil==2.8.1', 'smdebug-rulesconfig==1.0.0', 'openai==1.12.0', 'certifi==2023.7.22', 'py4j==0.10.9.5', 'types-pytz==2023.3.1.1', 'filelock==3.13.0', 'pyarrow==14.0.2', 'lark==1.1.7', 'opentelemetry-exporter-otlp-proto-http==1.22.0', 'partd==1.4.0', 'kafka-python==2.0.1', 'kr8s==0.13.1', 'fabric==3.2.1', 'msrest==0.6.20', 'datadog-api-client==2.21.0', 'kfp-server-api==2.0.3', 'ujson==5.7.0', 'elasticsearch==8.12.0', 'azure-batch==14.1.0', 'fsspec==2023.12.1', 'pendulum==2.1.1', 'apache-airflow-providers-common-sql==1.10.0', 'azure-mgmt-recoveryservicesbackup==9.0.0', 'google-re2==1.1', 'flask-cors==3.0.10', 'jsonpickle==3.0.2', 'ecdsa==0.17.0', 'google-cloud-kms==2.21.1', 'google-cloud-dlp==3.15.2', 'gradio-client==0.9.0', 'funcsigs==1.0.0', 'celery==5.3.5', 'bitarray==2.9.1', 'pyspark==3.4.2', 'types-setuptools==69.1.0.20240217', 'google-cloud-tasks==2.16.1', 'makefun==1.15.2', 'google-pasta==0.1.8', 'opentelemetry-api==1.21.0', 'semantic-version==2.8.5', 'zipp==3.17.0', 'mergedeep==1.3.4', 'mako==1.3.2', 'marshmallow==3.20.2', 'boto3==1.34.55', 'logbook==1.6.0', 'pycryptodomex==3.19.1', 'pyflakes==3.1.0', 'azure-mgmt-sqlvirtualmachine==0.4.0', 'transformers==4.38.0', 'jsonpickle==3.0.0', 'mkdocs-material==9.5.11', 'pep517==0.13.1', 'pytest-runner==6.0.0', 'toml==0.10.1', 'django-extensions==3.2.1', 'sphinx==7.2.4', 'gensim==4.3.0', 'opencensus-ext-azure==1.1.12', 'tabulate==0.8.9', 'asyncache==0.3.1', 'azure-keyvault==4.0.0', 'hvac==2.0.0', 'evidently==0.4.14', 'uamqp==1.6.6', 'scipy==1.12.0', 'tensorflow-io-gcs-filesystem==0.36.0', 'keras-preprocessing==1.1.0', 'google-pasta==0.2.0', 'pathy==0.10.2', 'checkov==3.2.24', 'marshmallow==3.21.0', 'hyperlink==20.0.1', 'docker-pycreds==0.3.0', 'json5==0.9.16', 'confection==0.1.3', 'gcsfs==2023.12.1', 'types-python-dateutil==2.8.19.14', 'flask-caching==2.1.0', 'azure-appconfiguration==1.5.0', 'delta-spark==3.0.0', 'types-awscrt==0.20.2', 'pycparser==2.21', 'nvidia-cublas-cu12==12.2.5.6', 'multidict==6.0.5', 'google-cloud-secret-manager==2.18.0', 'aiobotocore==2.11.2', 'h3==3.7.4', 'mypy-boto3-appflow==1.34.0', 'tqdm==4.66.2', 'google-cloud-build==3.23.2', 'fastapi==0.109.1', 'azure-mgmt-containerservice==29.0.0', 'azure-mgmt-compute==30.3.0', 'more-itertools==10.0.0', 'opt-einsum==3.2.1', 'apscheduler==3.10.2', 'omegaconf==2.2.3', 'aws-psycopg2==1.2.1', 'torch==2.2.1', 'typeguard==4.1.5', 'shap==0.43.0', 'python-multipart==0.0.9', 'djangorestframework==3.14.0', 'pathos==0.3.2', 'libcst==1.2.0', 'nvidia-cudnn-cu11==8.9.4.25', 'sqlalchemy==2.0.27', 'urllib3==2.2.0', 'grpcio-tools==1.60.0', 'google-cloud-resource-manager==1.12.1', 'tensorstore==0.1.52', 'regex==2023.12.25', 'h5py==3.8.0', 'aiodns==3.1.1', 'azure-mgmt-recoveryservices==2.3.0', 'more-itertools==10.1.0', 'simplejson==3.19.1', 'azure-mgmt-resource==22.0.0', 'openapi-spec-validator==0.6.0', 'imagesize==1.4.1', 'zipp==3.16.1', 'dm-tree==0.1.6', 'json5==0.9.17', 'flask-sqlalchemy==3.1.1', 'pyjwt==2.6.0', 'pg8000==1.30.5', 'mashumaro==3.12', 'arrow==1.2.3', 'pyhcl==0.4.5', 'db-contrib-tool==0.6.14', 'pyhcl==0.4.4', 'jax==0.4.22', 'readme-renderer==42.0', 'msgpack==1.0.8', 'azure-mgmt-msi==7.0.0', 'email-validator==2.1.1', 'opencensus==0.11.4', 'distrax==0.1.4', 'linkify-it-py==2.0.1', 'multidict==6.0.3', 'readme-renderer==43.0', 'magicattr==0.1.6', 'databricks-api==0.7.0', 'identify==2.5.33', 'azure-cli-core==2.55.0', 'httptools==0.6.1', 'numexpr==2.8.8', 'decorator==5.1.0', 'typing-extensions==4.10.0', 'wcwidth==0.2.11', 'jaxlib==0.4.23', 'jpype1==1.4.1', 'pybind11==2.11.1', 'sortedcontainers==2.2.2', 'absl-py==2.1.0', 'sshtunnel==0.4.0', 'alembic==1.13.1', 'azure-kusto-data==4.2.0', 'typed-ast==1.5.3', 'cattrs==23.2.2', 'google-cloud-storage==2.15.0', 'pyperclip==1.8.0', 'nbformat==5.9.2', 'starkbank-ecdsa==2.2.0', 'marshmallow==3.20.0', 'evergreen-py==3.6.21', 'requests-aws4auth==1.2.1', 'junit-xml==1.8', 'watchtower==2.1.1', 'docstring-parser==0.15', 'holidays==0.44', 'imdbpy==2021.4.18', 'pandas-gbq==0.19.2', 'nvidia-cuda-nvrtc-cu12==12.3.52', 'jupyter-events==0.7.0', 'apache-beam==2.54.0', 'graphql-core==3.2.1', 'azure-mgmt-media==10.2.0', 'alembic==1.13.0', 'croniter==2.0.0', 'async-timeout==4.0.3', 'google-resumable-media==2.5.0', 'pydot==1.4.2', 'pynacl==1.5.0', 'universal-pathlib==0.2.1', 'checkov==3.2.31', 'anyio==4.2.0', 'markupsafe==2.1.4', 'uri-template==1.3.0', 'django==4.2.9', 'cloudpathlib==0.17.0', 'inflect==6.1.1', 'watchtower==3.0.0', 'ecdsa==0.18.0', 'types-s3transfer==0.9.0', 'office365-rest-python-client==2.5.3', 'kr8s==0.13.2', 'google==2.0.3', 'azure-mgmt-iothubprovisioningservices==0.3.0', 'marshmallow-sqlalchemy==0.30.0', 'entrypoints==0.2.3', 'fastjsonschema==2.19.0', 'grpcio-tools==1.60.1', 'django-cors-headers==4.3.0', 'ddtrace==2.6.3', 'botocore-stubs==1.34.55', 'evaluate==0.4.1', 'pymeeus==0.5.11', 'gitpython==3.1.42', 'azure-mgmt-web==7.1.0', 'coloredlogs==15.0', 'typing-inspect==0.9.0', 'joblib==1.3.0', 'astunparse==1.6.2', 'installer==0.6.0', 'configargparse==1.5.3', 'websocket-client==1.7.0', 'argcomplete==3.2.0', 'humanfriendly==9.2', 'asttokens==2.4.0', 'thrift==0.16.0', 'fire==0.4.0', 'cligj==0.7.0', 'pycryptodome==3.20.0', 'kr8s==0.13.5', 'texttable==1.7.0', 'types-setuptools==69.1.0.20240229', 'mmh3==4.0.0', 'rich-argparse==1.2.0', 'mdit-py-plugins==0.4.0', 'watchdog==2.3.1', 'bokeh==3.3.3', 'netaddr==1.2.1', 'libclang==16.0.6', 'cachecontrol==0.14.0', 'rfc3986==2.0.0', 'querystring-parser==1.2.2', 'humanize==4.8.0', 'aenum==3.1.13', 'mlflow==2.11.0', 'jupyter-core==5.6.1', 'nose==1.3.6', 'tenacity==8.2.3', 'gspread==6.0.1', 'google-cloud-aiplatform==1.42.0', 'azure-synapse-accesscontrol==0.5.0', 'google-cloud-firestore==2.13.1', 'python-multipart==0.0.7', 'xgboost==2.0.2', 'pycryptodome==3.19.0', 'google-cloud-language==2.13.1', 'watchfiles==0.21.0', 'referencing==0.32.0', 'parse-type==0.6.0', 'cligj==0.7.2', 'spacy-loggers==1.0.3', 'psycopg2-binary==2.9.8', 'datasets==2.16.1', 'uritemplate==4.0.0', 'cloudpathlib==0.16.0', 'cython==3.0.7', 'azure-mgmt-msi==6.0.1', 'wandb==0.16.1', 'webdriver-manager==4.0.1', 'apscheduler==3.10.4', 'paramiko==3.3.0', 'nbformat==5.9.0', 'fire==0.3.1', 'geographiclib==2.0', 'jaydebeapi==1.2.1', 'httpcore==1.0.2', 'jupyterlab-server==2.25.1', 'ec2-metadata==2.12.0', 'moto==5.0.0', 'azure-mgmt-batch==17.2.0', 'azure-mgmt-appconfiguration==2.1.0', 'aiosignal==1.2.0', 'python-daemon==2.3.2', 'magicattr==0.1.4', 'azure-mgmt-core==1.3.2', 'resolvelib==1.0.0', 'joblib==1.3.1', 'cffi==1.15.1', 'requests-file==1.5.1', 'azure-keyvault-administration==4.4.0', 'connexion==3.0.6', 'google-cloud-datastore==2.19.0', 'mergedeep==1.3.2', 'python-jsonpath==1.0.0', 'rfc3986==1.4.0', 'pytest-forked==1.5.0', 'mdit-py-plugins==0.3.4', 'tensorstore==0.1.54', 'types-pyyaml==6.0.12.12', 'ndg-httpsclient==0.4.4', 'kubernetes==27.2.0', 'mccabe==0.6.1', 'jupyter-client==8.6.0', 'geopandas==0.14.2', 'six==1.15.0', 'pypdf2==3.0.0', 'tiktoken==0.5.1', 'universal-pathlib==0.1.4', 'azure-mgmt-resource==23.0.0', 'geopy==2.4.0', 'mistune==3.0.0', 'multiprocess==0.70.14', 'cachelib==0.10.2', 'invoke==2.1.2', 'unidecode==1.3.6', 'six==1.16.0', 'oauthlib==3.2.0', 'onnx==1.14.0', 'starkbank-ecdsa==2.1.0', 'pycountry==23.12.11', 'tomli==2.0.0', 'aws-sam-translator==1.84.0', 'datetime==5.3', 'secretstorage==3.3.3', 'mkdocs-material==9.5.9', 'azure-synapse-artifacts==0.18.0', 'termcolor==2.2.0', 'pyjwt==2.8.0', 'wcwidth==0.2.12', 'azure-keyvault==4.1.0', 'tldextract==5.0.1', 'robotframework-seleniumlibrary==6.2.0', 'pytest-html==4.1.1', 'databricks-sdk==0.19.0', 'dpath==2.1.4', 'cmake==3.28.1', 'srsly==2.4.6', 'importlib-metadata==6.11.0', 'retry==0.9.2', 'argparse==1.4.0', 'google-cloud-bigquery-storage==2.24.0', 'rapidfuzz==3.5.2', 'azure-mgmt-resource==23.0.1', 'nvidia-nccl-cu12==2.18.1', 'opencv-python==4.8.1.78', 'pathlib2==2.3.7', 'orbax-checkpoint==0.5.1', 'cython==3.0.6', 'authlib==1.3.0', 'azure-cli==2.56.0', 'types-pyyaml==6.0.12.11', 'hpack==3.0.0', 'matplotlib-inline==0.1.6', 'nvidia-cufft-cu12==11.0.8.103', 'wandb==0.16.3', 'looker-sdk==24.0.0', 'loguru==0.7.2', 'snowballstemmer==2.2.0', 'fiona==1.9.4', 'h5py==3.9.0', 'nest-asyncio==1.5.8', 'google-ads==23.0.0', 'google-cloud-appengine-logging==1.4.1', 'requests==2.30.0', 'hyperlink==20.0.0', 'oscrypto==1.2.0', 'argcomplete==3.2.2', 'cloudpickle==3.0.0', 'azure-cli-core==2.57.0', 'keras-applications==1.0.7', 'unittest-xml-reporting==3.1.0', 'setuptools-scm==8.0.2', 'grpc-google-iam-v1==0.12.6', 'sentence-transformers==2.3.1', 'pipenv==2023.12.0', 'lazy-object-proxy==1.8.0', 'pip==23.3.2', 'wrapt==1.15.0', 'coverage==7.4.1', 'newrelic==9.6.0', 'cymem==2.0.7', 'tensorflow-io==0.35.0', 'numpy==1.26.4', 'locket==1.0.0', 'imageio==2.33.0', 'spacy==3.7.2', 'avro==1.11.2', 'hpack==2.3.0', 'altair==5.1.2', 'pandas==2.1.4', 'pip-tools==7.3.0', 'jeepney==0.7.1', 'uvicorn==0.27.1', 'cron-descriptor==1.4.3', 'sagemaker==2.207.1', 'statsmodels==0.13.5', 'azure-mgmt-applicationinsights==3.1.0', 'absl-py==1.4.0', 'aiobotocore==2.11.0', 'prettytable==3.10.0', 'virtualenv==20.25.1', 'mdit-py-plugins==0.3.5', 'azure-mgmt-containerservice==29.1.0', 'boto==2.48.0', 'cachetools==5.3.1', 'azure-mgmt-reservations==2.2.0', 'azure-mgmt-policyinsights==0.6.0', 'types-protobuf==4.24.0.20240302', 'botocore-stubs==1.34.54', 'convertdate==2.3.2', 'cookiecutter==2.5.0', 'tzdata==2023.3', 'pathy==0.10.3', 'fasteners==0.18', 'rdflib==6.3.2', 'promise==2.2.1', 'mypy-boto3-rds==1.34.30', 'frozendict==2.4.0', 'sqlalchemy-jsonfield==1.0.0', 'importlib-resources==6.1.0', 'trio-websocket==0.11.1', 'types-redis==4.6.0.11', 'tableauserverclient==0.28', 'freezegun==1.3.1', 'backoff==2.2.1', 'sphinxcontrib-serializinghtml==1.1.9', 'configargparse==1.7', 'pendulum==3.0.0', 'types-redis==4.6.0.20240106', 'notebook-shim==0.2.2', 'hdfs==2.7.1', 's3transfer==0.10.0', 'selenium==4.18.1', 'azure-mgmt-botservice==1.0.0', 'langsmith==0.1.8', 'javaproperties==0.8.0', 'langsmith==0.1.17', 'toolz==0.12.0', 'google-cloud-aiplatform==1.42.1', 'python-gnupg==0.5.2', 'email-validator==1.3.1', 'protobuf3-to-dict==0.1.5', 'azure-synapse-spark==0.6.0', 'shortuuid==1.0.9', 'pytzdata==2019.3', 'cachetools==5.3.3', 'google-cloud-language==2.13.0', 'great-expectations==0.18.10', 'pyhive==0.7.0', 'alembic==1.12.1', 'libcst==1.1.0', 'python-dateutil==2.8.0', 'azure-appconfiguration==1.4.0', 'python-dateutil==2.9.0', 'protobuf3-to-dict==0.1.4', 'azure-mgmt-network==25.3.0', 'pip==24.0', 'flask-appbuilder==4.4.0', 'grpcio==1.62.0', 'inflection==0.4.0', 'azure-mgmt-redis==14.3.0', 'notebook-shim==0.2.4', 'appdirs==1.4.2', 'tensorflow-io-gcs-filesystem==0.34.0', 'tldextract==5.1.1', 'elasticsearch-dsl==8.11.0', 'isort==5.13.0', 'numpy==1.26.3', 'opensearch-py==2.4.2', 'pyelftools==0.30', 'colorlog==6.7.0', 'azure-mgmt-marketplaceordering==1.1.0', 'argparse==1.2.2', 'opt-einsum==3.2.0', 'azure-mgmt-core==1.4.0', 'stevedore==5.2.0', 'multiprocess==0.70.16', 'huggingface-hub==0.20.1', 'azure-graphrbac==0.61.0', 'pyhive==0.6.4', 'querystring-parser==1.2.3', 'gitdb==4.0.11', 'pyopenssl==24.0.0', 'xarray==2024.2.0', 'constructs==10.2.69', 'jsonlines==4.0.0', 'click-man==0.3.0', 'cffi==1.16.0', 'bytecode==0.14.2', 'dbt-postgres==1.7.6', 'enum34==1.1.10', 'nvidia-cusparse-cu12==12.1.3.153', 'itsdangerous==2.1.0', 'dnspython==2.6.0', 'future==1.0.0', 'trove-classifiers==2024.3.3', 'configparser==6.0.1', 'geopandas==0.14.3', 'fqdn==1.4.0', 'factory-boy==3.2.0', 'azure-mgmt-maps==2.1.0', 'semver==3.0.1', 'google-cloud-container==2.42.0', 'schema==0.7.2', 'toml==0.10.2', 'azure-cosmos==4.5.1', 'zstandard==0.21.0', 'aenum==3.1.15', 'awscli==1.32.53', 'azure-storage-file-datalake==12.13.1', 'slicer==0.0.6', 'cookiecutter==2.6.0', 'python-gnupg==0.5.0', 'pydantic==2.6.1', 'defusedxml==0.6.0', 'psycopg2-binary==2.9.9', 'pyhive==0.6.5', 'keyring==24.2.0', 'rsa==4.8', 'apache-airflow-providers-ssh==3.10.1', 'google-api-python-client==2.117.0', 'google-cloud-dataproc==5.9.2', 'imagesize==1.3.0', 'llvmlite==0.41.1', 'pyspnego==0.9.2', 'defusedxml==0.7.0', 'typing-inspect==0.7.1', 'azure-common==1.1.28', 'azure-mgmt-recoveryservices==2.4.0', 'connexion==3.0.4', 'azure-keyvault-certificates==4.6.0', 'pydot==1.4.1', 'boto3-stubs==1.34.53', 'slackclient==2.9.4', 'astroid==3.0.2', 'pep517==0.12.0', 'greenlet==3.0.1', 'dask==2024.2.1', 'twisted==23.10.0', 'twine==4.0.1', 'prison==0.2.0', 'azure-keyvault-certificates==4.7.0', 'opentelemetry-exporter-otlp-proto-grpc==1.23.0', 'unidecode==1.3.8', 'azure-eventgrid==4.15.0', 'azure-mgmt-advisor==9.0.0', 'pluggy==1.2.0', 'pytorch-lightning==2.1.3', 'python-editor==1.0.4', 'astunparse==1.6.3', 'billiard==4.0.2', 'maxminddb==2.5.0', 'jira==3.6.0', 'google-cloud-bigtable==2.22.0', 'flit-core==3.7.1', 'structlog==23.2.0', 'jpype1==1.4.0', 'shapely==2.0.1', 'pysftp==0.2.9', 'cachecontrol==0.12.14', 'httpx==0.27.0', 'pkginfo==1.9.6', 'pyodbc==5.1.0', 'editables==0.4', 'jaydebeapi==1.2.2', 'threadpoolctl==3.3.0', 'google-auth-httplib2==0.1.1', 'azure-mgmt-network==25.1.0', 'mako==1.3.0', 'jupyterlab-server==2.25.2', 'cssselect==1.2.0', 'rich==13.7.1', 'tensorflow-estimator==2.13.0', 'cymem==2.0.8', 'parse==1.20.0', 'json5==0.9.19', 'chardet==5.0.0', 'gradio-client==0.10.0', 'asyncio==3.4.3', 'nvidia-cudnn-cu11==8.9.6.50', 'gitpython==3.1.41', 'google-cloud-vision==3.7.1', 'distrax==0.1.5', 'azure-mgmt-containerinstance==10.1.0', 'jupyterlab==4.1.3', 'structlog==23.3.0', 'tensorboard-data-server==0.7.2', 'sphinxcontrib-devhelp==1.0.6', 'evaluate==0.4.0', 'charset-normalizer==3.3.0', 'resolvelib==0.9.0', 'jupyter-server-terminals==0.5.2', 'webencodings==0.5.1', 'termcolor==2.3.0', 'nvidia-nvtx-cu12==12.3.52', 'python-editor==1.0.3', 'humanize==4.9.0', 'robotframework-seleniumlibrary==6.1.2', 'lightgbm==4.3.0', 'more-itertools==10.2.0', 'onnx==1.14.1', 'pycountry==22.3.5', 'croniter==1.4.1', 'imageio==2.34.0', 'markdown==3.5.1', 'dask==2024.2.0', 'emoji==2.10.1', 'ruamel-yaml-clib==0.2.7', 'nbformat==5.9.1', 'google==2.0.2', 'google-auth-httplib2==0.2.0', 'reportlab==4.0.9', 'gevent==23.9.0', 'azure-common==1.1.27', 'geopandas==0.14.1', 'iso8601==1.1.0', 'cog==0.9.2', 'click-repl==0.2.0', 'asynctest==0.12.3', 'user-agents==2.0', 'text-unidecode==1.1', 'flax==0.8.0', 'opentelemetry-exporter-otlp-proto-common==1.22.0', 'azure-mgmt-keyvault==10.3.0', 'dill==0.3.7', 'pyparsing==3.1.0', 'smart-open==6.3.0', 'lz4==4.3.3', 'azure-mgmt-keyvault==10.2.2', 'sympy==1.11', 'pywavelets==1.3.0', 'azure-mgmt-eventgrid==10.0.0', 'tzdata==2024.1', 'avro-python3==1.10.0', 'flask-babel==3.1.0', 'rich-argparse==1.4.0', 'aiosignal==1.3.1', 'scandir==1.10.0', 'azure-identity==1.14.1', 'mypy-boto3-s3==1.34.0', 'fonttools==4.48.1', 'gitdb==4.0.10', 'ppft==1.7.6.7', 'nvidia-cusolver-cu12==11.5.4.101', 'debugpy==1.8.0', 'dbt-extractor==0.5.0', 'markupsafe==2.1.3', 'google-api-core==2.16.2', 'starlette==0.37.0', 'pydantic==2.6.3', 'cerberus==1.3.5', 'azure-mgmt-iothub==2.3.0', 'gsutil==5.27', 'plotly==5.17.0', 'google-cloud-audit-log==0.2.3', 'kfp-pipeline-spec==0.3.0', 'azure-mgmt-datamigration==4.1.0', 'jupyter-console==6.6.3', 'kfp==2.7.0', 'streamlit==1.30.0', 'slack-sdk==3.26.1', 'flit-core==3.8.0', 'dacite==1.8.1', 'libcst==1.0.1', 'awscli==1.32.49', 'flask-session==0.6.0', 'sentence-transformers==2.4.0', 'avro==1.11.1', 'requests-ntlm==1.2.0', 'einops==0.6.0', 'nvidia-cudnn-cu11==8.9.5.30', 'promise==2.2', 'azure-datalake-store==0.0.51', 'mypy-extensions==0.4.4', 'graphviz==0.19.2', 'xlsxwriter==3.1.9', 'google-cloud-aiplatform==1.43.0', 'delta-spark==2.4.0', 'setuptools-scm==8.0.4', 'python-gitlab==4.2.0', 'docstring-parser==0.14.1', 'looker-sdk==24.2.0', 'watchdog==3.0.0', 'opencv-python-headless==4.8.0.76', 'kornia==0.7.1', 'azure-core==1.29.7', 'apache-airflow-providers-snowflake==5.3.0', 'hologram==0.0.16', 'msrest==0.6.21', 'wsproto==1.0.0', 'confection==0.1.4', 'opentelemetry-sdk==1.23.0', 'protobuf==4.25.1', 'py4j==0.10.9.6', 'azure-mgmt-servicebus==8.2.0', 'langcodes==3.3.0', 'llvmlite==0.42.0', 'polars==0.20.8', 'stack-data==0.6.3', 'db-contrib-tool==0.6.13', 'ddtrace==2.6.4', 'feedparser==6.0.11', 'ua-parser==0.16.0', 'sentencepiece==0.2.0', 'tblib==3.0.0', 'yapf==0.32.0', 'partd==1.4.1', 'sentence-transformers==2.5.1', 'cron-descriptor==1.3.0', 'firebase-admin==6.2.0', 'fuzzywuzzy==0.17.0', 'pybind11==2.10.4', 'opencv-python==4.9.0.80', 'xmltodict==0.11.0', 'azure-mgmt-iothubprovisioningservices==1.1.0', 'azure-keyvault-certificates==4.8.0', 'pytest-asyncio==0.23.4', 'cerberus==1.3.3', 'transformers==4.37.2', 'slicer==0.0.7', 'opencv-python==4.8.0.76', 'ipaddress==1.0.21', 'pypdf2==3.0.1', 'nvidia-cuda-nvrtc-cu12==12.3.107']
    packages_ll["data_4"]=packages_l
    n_samples_d["data_4"]=4
    test_portion_d["data_4"]=0.25

    # highlight_label_set = set(['pluggy', 'docutils', 'lxml', 'azure-core', 'multidict', 'pyopenssl', 'greenlet', 'google-cloud-core', 'et-xmlfile', 'coverage', 'google-cloud-storage', 'openpyxl', 'google-api-python-client', 'rpds-py', 'asn1crypto', 'bcrypt', 'itsdangerous', 'google-resumable-media', 'pynacl', 'google-cloud-bigquery', 'pathspec', 'regex', 'joblib', 'cython', 'mdit-py-plugins', 'sagemaker', 'smmap', 'mypy-extensions', 'msgpack', 'ptyprocess', 'azure-common', 'msrest', 'future', 'dnspython', 'py', 'snowflake-connector-python', 'portalocker', 'py4j', 'keyring', 'google-crc32c', 'awswrangler', 'fonttools', 'markdown-it-py', 'kiwisolver', 'azure-identity', 'xmltodict', 'threadpoolctl', 'ipython', 'backoff', 'poetry-plugin-export', 'google-auth-httplib2', 'sortedcontainers', 'oscrypto', 'nest-asyncio', 'mccabe', 'redshift-connector', 'mako', 'pkgutil-resolve-name', 'traitlets', 'pyodbc', 'black', 'typing-inspect', 'datadog', 'jsonpointer', 'argcomplete', 'defusedxml', 'pymongo', 'google-cloud-pubsub', 'xlrd', 'poetry', 'cfn-lint', 'requests-aws4auth', 'parso', 'jsonpath-ng', 'contourpy', 'python-json-logger', 'pydantic-core', 'fastjsonschema', 'backcall', 'notebook', 'astroid', 'nbformat', 'rapidfuzz', 'matplotlib-inline', 'tensorflow', 'pylint', 'transformers', 'setuptools-scm', 'h5py', 'kubernetes', 'jsonpatch', 'huggingface-hub', 'imageio', 'grpc-google-iam-v1', 'annotated-types', 'debugpy', 'entrypoints', 'smart-open', 'llvmlite', 'msrestazure', 'numba', 'dulwich', 'google-cloud-secret-manager', 'elasticsearch', 'tensorflow-estimator', 'lockfile', 'aiofiles', 'orjson', 'great-expectations', 'aenum', 'mypy', 'pygithub', 'requests-file', 'cleo', 'nodeenv', 'gast', 'identify', 'comm', 'nbclient', 'tokenizers', 'django', 'send2trash', 'cached-property', 'deepdiff', 'croniter', 'ipywidgets', 'execnet', 'overrides', 'widgetsnbextension', 'jupyterlab-server', 'jupyterlab', 'keras', 'typer', 'hvac', 'dataclasses', 'cfgv', 'asttokens', 'aws-sam-translator', 'selenium', 'distro', 'typeguard', 'executing', 'stack-data', 'xgboost', 'confluent-kafka', 'rfc3986', 'pure-eval', 'tblib', 'apache-airflow', 'fastavro', 'uri-template', 'db-dtypes', 'unidecode', 'prettytable', 'docopt', 'retrying', 'libclang', 'thrift', 'pymssql', 'zeep', 'rfc3986-validator', 'argon2-cffi-bindings', 'inflection', 'jupyter-lsp', 'openai', 'moto', 'opentelemetry-proto', 'snowballstemmer', 'ujson', 'sphinxcontrib-qthelp', 'sphinxcontrib-devhelp', 'azure-graphrbac', 'sphinxcontrib-applehelp', 'aioconsole', 'python-gnupg', 'parsedatetime', 'google-cloud-firestore', 'pox', 'pathos', 'libcst', 'kombu', 'applicationinsights', 'shap', 'zope-event', 'numexpr', 'trio', 'argparse', 'gevent', 'email-validator', 'torchvision', 'google-cloud-appengine-logging', 'kafka-python', 'checkov', 'tensorboard-plugin-wit', 'coloredlogs', 'apache-beam', 'azure-mgmt-storage', 'tldextract', 'colorlog', 'wandb', 'azure-eventhub', 'pywavelets', 'docstring-parser', 'datetime', 'fire', 'makefun', 'google-cloud-resource-manager', 'uamqp', 'ecdsa', 'slicer', 'hpack', 'imagesize', 'google-cloud-logging', 'wsproto', 'delta-spark', 'validators', 'fiona', 'databricks-sql-connector', 'sshtunnel', 'brotli', 'holidays', 'apache-airflow-providers-common-sql'])
    # highlight_label_set = set(['platformdirs', 'certifi', 'jmespath', 'aiohttp', 'async-timeout', 'pyparsing', 'pydantic', 'importlib-resources', 'websocket-client', 'aiosignal', 'distlib', 'gitpython', 'tabulate', 'proto-plus', 'msal', 'azure-storage-blob', 'tzlocal', 'docker', 'grpcio-tools', 'sqlparse', 'wcwidth', 'poetry-core', 'sniffio', 'google-auth-oauthlib', 'jaraco-classes', 'dill', 'alembic', 'httplib2', 'python-dotenv', 'scramp', 'tb-nightly', 'marshmallow', 'uritemplate', 'toml', 'trove-classifiers', 'cycler', 'jeepney', 'pyzmq', 'toolz', 'prometheus-client', 'httpcore', 'adal', 'shellingham', 'pyflakes', 'httpx', 'pkginfo', 'sentry-sdk', 'nbconvert', 'fastapi', 'flake8', 'python-utils', 'asynctest', 'google-cloud-bigquery-storage', 'databricks-cli', 'starlette', 'aioitertools', 'pickleshare', 'mistune', 'jupyter-server', 'pbr', 'ipykernel', 'build', 'arrow', 'asgiref', 'uvicorn', 'html5lib', 'pyproject-hooks', 'oauth2client', 'tinycss2', 'altair', 'multiprocess', 'zope-interface', 'retry', 'crashtest', 'httptools', 'querystring-parser', 'contextlib2', 'tensorboard-data-server', 'azure-storage-file-datalake', 'xlsxwriter', 'configparser', 'mysql-connector-python', 'pendulum', 'text-unidecode', 'semver', 'responses', 'pipenv', 'snowflake-sqlalchemy', 'python-slugify', 'pytest-xdist', 'sphinx', 'jupyterlab-widgets', 'gremlinpython', 'click-plugins', 'pytest-mock', 'azure-storage-common', 'dataclasses-json', 'futures', 'pandocfilters', 'patsy', 'xxhash', 'tensorflow-io-gcs-filesystem', 'jupyterlab-pygments', 'setproctitle', 'astunparse', 'async-lru', 'gcsfs', 'azure-keyvault-secrets', 'pysftp', 'ordered-set', 'faker', 'semantic-version', 'jsonpickle', 'pytest-runner', 'sphinxcontrib-serializinghtml', 'webcolors', 'azure-datalake-store', 'typing', 'isoduration', 'jupyter-server-terminals', 'deprecation', 'opencensus-context', 'typed-ast', 'opencensus', 'stevedore', 'pyproj', 'gspread', 'ppft', 'watchtower', 'trio-websocket', 'azure-mgmt-keyvault', 'structlog', 'opentelemetry-exporter-otlp-proto-http', 'opentelemetry-semantic-conventions', 'enum34', 'pathlib2', 'types-urllib3', 'pybind11', 'pydata-google-auth', 'lightgbm', 'opencensus-ext-azure', 'lz4', 'cligj', 'azure-mgmt-containerregistry', 'keras-preprocessing', 'unittest-xml-reporting', 'partd', 'schema', 'flask-cors', 'alabaster', 'azure-mgmt-authorization', 'h2', 'python-http-client', 'amqp', 'pytest-asyncio', 'locket', 'hyperframe'])
    # highlight_label_set = set(['platformdirs', 'certifi', 'jmespath', 'packaging', 'numpy', 'googleapis-common-protos', 'aiohttp', 'filelock', 'async-timeout', 'pyparsing', 'pydantic', 'tqdm', 'importlib-resources', 'decorator', 'pygments', 'websocket-client', 'pymysql', 'aiosignal', 'distlib', 'gitpython', 'tomlkit', 'paramiko', 'tabulate', 'iniconfig', 'proto-plus', 'requests-toolbelt', 'msal', 'psycopg2-binary', 'azure-storage-blob', 'anyio', 'tzlocal', 'docker', 'grpcio-tools', 'sqlparse', 'gitdb', 'wcwidth', 'poetry-core', 'sniffio', 'google-auth-oauthlib', 'pexpect', 'jaraco-classes', 'pyrsistent', 'markdown', 'ruamel-yaml', 'tornado', 'prompt-toolkit', 'dill', 'alembic', 'tenacity', 'cloudpickle', 'gunicorn', 'tzdata', 'httplib2', 'rich', 'msal-extensions', 'python-dotenv', 'scramp', 'tb-nightly', 'marshmallow', 'uritemplate', 'webencodings', 'cachecontrol', 'h11', 'toml', 'jedi', 'trove-classifiers', 'pycryptodome', 'cycler', 'pg8000', 'ply', 'jeepney', 'pyzmq', 'toolz', 'prometheus-client', 'isort', 'secretstorage', 'httpcore', 'adal', 'shellingham', 'pyflakes', 'jupyter-core', 'tensorboard', 'httpx', 'pkginfo', 'sentry-sdk', 'nbconvert', 'fastapi', 'mdurl', 'flake8', 'python-utils', 'asynctest', 'google-cloud-bigquery-storage', 'databricks-cli', 'progressbar2', 'starlette', 'aioitertools', 'pickleshare', 'mistune', 'jupyter-server', 'lazy-object-proxy', 'pbr', 'appdirs', 'ipykernel', 'build', 'arrow', 'bleach', 'asgiref', 'uvicorn', 'html5lib', 'websockets', 'pyproject-hooks', 'cattrs', 'mlflow', 'oauth2client', 'tinycss2', 'altair', 'multiprocess', 'zope-interface', 'mock', 'retry', 'google-pasta', 'flatbuffers', 'crashtest', 'pysocks', 'shapely', 'httptools', 'querystring-parser', 'contextlib2', 'tensorboard-data-server', 'azure-storage-file-datalake', 'xlsxwriter', 'notebook-shim', 'tox', 'configparser', 'mysql-connector-python', 'pendulum', 'text-unidecode', 'json5', 'pre-commit', 'semver', 'responses', 'pipenv', 'installer', 'snowflake-sqlalchemy', 'pytzdata', 'argon2-cffi', 'python-slugify', 'pytest-xdist', 'sphinx', 'jupyterlab-widgets', 'jupyter-events', 'gremlinpython', 'azure-mgmt-core', 'click-plugins', 'pytest-mock', 'azure-storage-common', 'slack-sdk', 'opentelemetry-sdk', 'tensorflow-serving-api', 'ipython-genutils', 'dataclasses-json', 'futures', 'pandocfilters', 'patsy', 'safetensors', 'xxhash', 'tensorflow-io-gcs-filesystem', 'opentelemetry-api', 'jupyterlab-pygments', 'setproctitle', 'astunparse', 'async-lru', 'terminado', 'gcsfs', 'azure-keyvault-secrets', 'pysftp', 'ordered-set', 'faker', 'semantic-version', 'jsonpickle', 'pytest-runner', 'sphinxcontrib-serializinghtml', 'webcolors', 'azure-datalake-store', 'antlr4-python3-runtime', 'typing', 'isoduration', 'jupyter-server-terminals', 'graphviz', 'fqdn', 'boto', 'deprecation', 'opencensus-context', 'typed-ast', 'opencensus', 'sphinxcontrib-htmlhelp', 'stevedore', 'pyproj', 'gspread', 'ppft', 'watchtower', 'trio-websocket', 'azure-mgmt-keyvault', 'structlog', 'opentelemetry-exporter-otlp-proto-http', 'simple-salesforce', 'celery', 'mypy-boto3-rds', 'prometheus-flask-exporter', 'opentelemetry-semantic-conventions', 'enum34', 'pathlib2', 'pycrypto', 'types-urllib3', 'pybind11', 'djangorestframework', 'pydata-google-auth', 'lightgbm', 'opencensus-ext-azure', 'azure-cosmos', 'lz4', 'cligj', 'pyhcl', 'azure-mgmt-containerregistry', 'python-jose', 'keras-preprocessing', 'unittest-xml-reporting', 'partd', 'schema', 'flask-cors', 'alabaster', 'azure-mgmt-authorization', 'h2', 'nvidia-cublas-cu11', 'python-http-client', 'amqp', 'hypothesis', 'pytest-asyncio', 'locket', 'hyperframe'])
    # highlight_label_set = set(['s3fs', 'cryptography', 'emoji'])
    # highlight_label_set = set(['cffi', 'PyJWT', 'attrs', 'pyasn1', 'click', 'pytz', 'MarkupSafe', 'grpcio-status', 'psutil', 'frozenlist', 'botocore', 'soupsieve', 'grpcio', 'awscli', 'yarl', 'idna', 'google-api-core', 'charset-normalizer', 'authlib', 'seaborn', 'colorama', 'pytest', 'NLTK', 'Flask', 'oauthlib', 'pycparser', 'nvidia-cuda-runtime-cu11', 'pandas', 'jinja2', 'scikit-learn', 'triton==2.0.0', 'deap', 'nvidia-cuda-nvrtc-cu11', 'cmake', 'astropy', 'bokeh', 'requests', 'biopython', 'redis', 'Scrapy', 'simplejson', 'opencv-python', 'opacus', 'scoop', 'plotly', 'Theano', 'mahotas', 'nilearn', 'beautifulsoup4', 'statsmodels', 'networkx', 's3transfer', 'scipy', 'SQLAlchemy', 'matplotlib', 'setuptools', 'rsa', 'urllib3', 'pillow', 'pyspark'])
    highlight_label_set = set()

    
    # all_samples_select_l = [str(sampleidx) for sampleidx in range(n_samples)]
    # all_samples_select_set = set(all_samples_select_l)
    # test_portion = 0.25
    # sample_step = int(len(all_samples_select_set)*test_portion)
    # test_sample_batchsets_l = [set(all_samples_select_l[i0:i0+sample_step]) for i0 in range(0,n_samples,sample_step)]
    # ###################### choose CV batch ######################
    # for test_sample_batch_idx, test_samples_select_set in enumerate(test_sample_batchsets_l):
    #     train_samples_select_set = all_samples_select_set - test_samples_select_set 
    for test_sample_batch_idx in [0]:   # this is the initial idx of a test batch, i.e., if test batch size is 4*0.25 = 1, `test_sample_batch_idx`` means pick the 0-index element as the test batch.
        # test_sample_batch_idx = 4
        # test_samples_select_set = test_sample_batchsets_l[test_sample_batch_idx]
        # train_samples_select_set = all_samples_select_set - test_samples_select_set
        # #############################################################

        for dataset in ["data_4"]:
            n_samples = n_samples_d[dataset]
            test_portion = test_portion_d[dataset]
            packages_l = packages_ll[dataset]
            if dataset not in []:
                packages_l = [package.replace("==", "_v").replace(".","_") for package in packages_l]
            # # #############################################################
            # # testing with filtered tagsetfilenames
            # with open("/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/filters/big_train_filtered_pair_tagsetfilenames_d_same_length", 'rb') as tf:
            #     labels_tagfiles_d = yaml.load(tf, Loader=yaml.Loader)
            #     packages_l = list(labels_tagfiles_d.keys())
            #     n_samples = len(labels_tagfiles_d[packages_l[0]])
            # # #############################################################
            for with_filter in [False]:
                if with_filter:
                    with open("/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/filters/big_train_tokenshares_filter_set", 'rb') as tf:
                        tokenshares_filter_set_d[dataset] = yaml.load(tf, Loader=yaml.Loader)
                    with open("/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/filters/big_train_tokennoises_filter_set", 'rb') as tf:
                        tokennoises_filter_set_d[dataset] = yaml.load(tf, Loader=yaml.Loader)
                    tokens_filter_set = tokenshares_filter_set_d[dataset].union(tokennoises_filter_set_d[dataset])
                else:
                    tokens_filter_set = set()

                # packages_l = list(set(packages_l)-highlight_label_set)
                # packages_l = list(highlight_label_set)
                # highlight_label_set = set()
                # packages_l = list([packages_l[0]])
                # print(packages_l)

                for n_jobs in [32]:
                    for n_models, test_batch_count in zip([5],[1,1,1,1]): #([50,25,20,15,10,5,1],[1,1,1,1,1,1,8]): # ([1,25,10],[8,1,1])
                        for n_estimators in [100]:
                            for depth in [1]:
                                for tree_method in["exact"]: # "exact","approx","hist"
                                    for max_bin in [1]:
                                        for input_size, dim_compact_factor in zip([None],[1,1,1,1,1]): # [None, 6832, 13664, 27329, 54659,109319],[1,1,1,1] # [None, 500, 1000, 5000, 10000, 15000],[1,1,1,1,1,1]
                                            random_instance = random.Random(4)
                                            for shuffle_idx in range(3):
                                                # sample labels per sub-model
                                                randomized_packages_l = random_instance.sample(packages_l, len(packages_l))
                                                package_subset, step = [], len(randomized_packages_l)//n_models+1
                                                for i in range(0, len(randomized_packages_l), step):
                                                    package_subset.append(set(randomized_packages_l[i:i+step]))

                                                # cross-validation for samples
                                                for i, train_subset in enumerate(package_subset):
                                                    train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/big_train/"
                                                    test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/big_train/" # Cross Validation: testing a portion of the SL dataset
                                                    
                                                    # ###########################################################################
                                                    # # if reading from the dir. When testing with filtered tagsetfilenames, this is not needed.
                                                    # labels_tagfiles_d = defaultdict(list)
                                                    # for tag_file in os.listdir(train_tags_path):
                                                    #     if (tag_file[-3:] == 'tag') and (tag_file[:-4].rsplit('-', 1)[0] in train_subset):
                                                    #         if len(labels_tagfiles_d[tag_file[:-4].rsplit('-', 1)[0]]) < n_samples:
                                                    #             labels_tagfiles_d[tag_file[:-4].rsplit('-', 1)[0]].append(tag_file)
                                                    # ###########################################################################

                                                    ###########################################################################
                                                    # if reading from the dir and load from data 4 which is generated with containers. When testing with filtered tagsetfilenames, this is not needed.
                                                    labels_tagfiles_d = defaultdict(list)
                                                    for tag_file in os.listdir(train_tags_path):
                                                        tag_file_splitred = tag_file.rsplit('.')
                                                        if (tag_file[-3:] == 'tag') and (tag_file_splitred[1] in train_subset):
                                                            if len(labels_tagfiles_d[tag_file_splitred[1]]) < n_samples:
                                                                labels_tagfiles_d[tag_file_splitred[1]].append(tag_file)
                                                    ###########################################################################
                                                                
                                                    test_tagfiles_set, train_tagfiles_set = set(), set()
                                                    for label, traintagfiles in labels_tagfiles_d.items():
                                                        if label in train_subset:
                                                            test_tagfiles_set.update(traintagfiles[int(test_sample_batch_idx*n_samples*test_portion): int((test_sample_batch_idx+1)*n_samples*test_portion)])
                                                            train_tagfiles_set.update(set(traintagfiles)-test_tagfiles_set)
                                                    if len(test_tagfiles_set)==0:
                                                        print("error: got empty test_tagfiles_set")
                                                        sys.exit(-1)
                                                    train_tag_files_l = list(train_tagfiles_set)
                                                    test_tag_files_l = list(test_tagfiles_set)
                                                    cwd  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+"removesharedornoisestags/"
                                                    run_init_train(train_tags_path, test_tags_path, cwd, train_tags_init_l=train_tag_files_l, test_tags_l=test_tag_files_l, n_jobs=n_jobs, n_estimators=n_estimators, tokens_filter_set=tokens_filter_set, test_batch_count=test_batch_count, input_size=input_size, compact_factor=dim_compact_factor, depth=depth, tree_method=tree_method)
                                                    # break


        ###################################
        # run_iter_train()


        # # ###################################
        # # run_pred()
        # # Testing the ML dataset
        # for dataset in ["data_3"]:
        #     for with_filter in [False]:
        #         for n_jobs in [32]:
        #             for clf_njobs in [32]:
        #                 for n_models, test_batch_count in zip([5],[1,1,1,1]): # zip([50,25,20,15,10,5,1],[1,1,1,1,1,1,8]):
        #                     for n_estimators in [100]:
        #                         for depth in [1]:
        #                             for tree_method in["exact"]: # "exact","approx","hist"
        #                                 for max_bin in [1]:
        #                                     for input_size, dim_compact_factor in zip([None],[1,1,1,1,1]):
        #                                         for shuffle_idx in range(3):

        #                                             clf_path = []
        #                                             for i in range(n_models):
        #                                                 clf_pathname = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(clf_njobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+"removesharedornoisestags/model_init.json"
        #                                                 if os.path.isfile(clf_pathname):
        #                                                     clf_path.append(clf_pathname)
        #                                             cwd = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+"removesharedornoisestags/"
        #                                             test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/big_ML_biased_test/"
        #                             #    run_init_train(train_tags_path, test_tags_path, cwd, n_jobs=n_jobs, n_estimators=n_estimators, train_packages_select_set=train_subset, test_packages_select_set=test_subset, input_size=input_size, depth=depth, tree_method=tree_method)
        #                                             run_pred(cwd, clf_path, test_tags_path, n_jobs=n_jobs, n_estimators=n_estimators, test_batch_count=test_batch_count, input_size=input_size, compact_factor=dim_compact_factor, depth=depth, tree_method=tree_method)











# ########## some debugging scripts.
# d = defaultdict(list)
# for test_tag_file in train_tag_files_l:
#     d[test_tag_file[:-4].rsplit("-",1)[0]].append(test_tag_file)
# for k,v in d.items():
#     # print(k,len(v))
#     if len(v) != 5:
#         print("!!!!!")
#         break

# aenum_v3_1_13.0.tag