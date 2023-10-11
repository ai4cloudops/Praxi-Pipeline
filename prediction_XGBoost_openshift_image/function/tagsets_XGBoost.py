import os, pickle, time, gc
import yaml, json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.datasets import make_multilabel_classification
import sklearn.metrics as metrics
from sklearn.preprocessing import Normalizer
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

def read_tokens(tags_path, tag_file, cwd, packages_select_set, inference_flag):
    logger = build_logger(tag_file, cwd+"logs/")
    try:
        ret = {}
        if(tag_file[-3:] == 'tag') and (tag_file[:-3].rsplit('-', 1)[0] in packages_select_set or packages_select_set == set()):
            with open(tags_path + tag_file, 'rb') as tf:
                # print(tag_file)
                # tagset_files.append(tag_file)
                ret["tag_file"] = tag_file
                local_all_tags_set = set()
                instance_feature_tags_d = defaultdict(int)
                tagset = yaml.load(tf, Loader=yaml.Loader)
                # tagset = json.load(tf)   

                # feature 
                for tag_vs_count in tagset['tags']:
                    k,v = tag_vs_count.split(":")
                    local_all_tags_set.add(k)
                    instance_feature_tags_d[k] += int(v)
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
        logger.info('%s', e)
    return ret

def tagsets_to_matrix(tags_path, index_tag_mapping_path=None, tag_index_mapping_path=None, index_label_mapping_path=None, label_index_mapping_path=None, cwd="/home/cc/Praxi-study/Praxi-Pipeline/prediction_base_image/model_testing_scripts/cwd/", train_flag=False, inference_flag=True, iter_flag=False, packages_select_set=set(), input_size=None):
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
    pool = mp.Pool(processes=mp.cpu_count())
    # pool = mp.Pool(processes=1)
    data_instance_d_l = [pool.apply_async(read_tokens, args=(tags_path, tag_file, cwd, packages_select_set, inference_flag)) for tag_file in tqdm(os.listdir(tags_path))]
    data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
    for data_instance_d in data_instance_d_l:
        if len(data_instance_d) ==4:
                tagset_files.append(data_instance_d['tag_file'])
                all_tags_set.update(data_instance_d['local_all_tags_set'])
                tags_by_instance_l.append(data_instance_d['instance_feature_tags_d'])
                all_label_set.update(data_instance_d['labels'])
                labels_by_instance_l.append(data_instance_d['labels'])

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
        
    # with open(cwd+'tagset_files.txt', 'w') as f:
    #     for line in tagset_files:
    #         f.write(f"{line}\n")

    # Sorting instances
    if not inference_flag:
        zipped = list(zip(tagset_files, tags_by_instance_l, labels_by_instance_l))
        zipped.sort(key=lambda x: x[0])
        tagset_files, tags_by_instance_l, labels_by_instance_l = zip(*zipped)
    else:
        zipped = list(zip(tagset_files, tags_by_instance_l))
        zipped.sort(key=lambda x: x[0])
        tagset_files, tags_by_instance_l = zip(*zipped)



#############
    # # Save tag:count in mapping format
    # with open(tags_path+"all_tags_set.obj","wb") as filehandler:
    #      pickle.dump(all_tags_set, filehandler)
    # with open(tags_path+"all_label_set.obj","wb") as filehandler:
    #      pickle.dump(all_label_set, filehandler)
    # with open(tags_path+"tags_by_instance_l.obj","wb") as filehandler:
    #      pickle.dump(tags_by_instance_l, filehandler)
    # with open(tags_path+"labels_by_instance_l.obj","wb") as filehandler:
    #      pickle.dump(labels_by_instance_l, filehandler)
    # with open(tags_path+"tagset_files.obj","wb") as filehandler:
    #      pickle.dump(tagset_files, filehandler)

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
            instance_row = np.zeros(len(all_tags_l))
            for tag_name,tag_count in instance_tags_d.items():
                if tag_name in tag_index_mapping:  # remove new tags unseen in mapping.
                    instance_row[tag_index_mapping[tag_name]] = tag_count
                else:
                    removed_tags_l.append(tag_name)
        else:
            instance_row = np.zeros(input_size)
            for tag_name,tag_count in instance_tags_d.items():
                if tag_name in tag_index_mapping:  # remove new tags unseen in mapping.
                    instance_row[tag_index_mapping[tag_name]%input_size] = tag_count
                else:
                    removed_tags_l.append(tag_name)
        # else:
        #     # instance_row = np.zeros(input_size)
        #     instance_row = np.random.randint(1000000000, size=input_size)
        #     # instance_row = np.array(range(input_size-1,-1,-1))
        instance_row_list.append(instance_row)
    # instance_row_list.extend(instance_row_list)
    feature_matrix = np.vstack(instance_row_list)
    del instance_row_list
    with open(cwd+'removed_tags_l', 'wb') as fp:
        pickle.dump(removed_tags_l, fp)
    with open(cwd+'removed_tags_l.txt', 'w') as f:
        for line in removed_tags_l:
            f.write(f"{line}\n")
    


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

def run_init_train(train_tags_init_path, test_tags_path, cwd, n_jobs=64, n_estimators=100, train_packages_select_set=set(), test_packages_select_set=set(), input_size=None, depth=1, tree_method="auto", max_bin=6):
    # train_tags_init_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_0&2/big_SL_biased_test/"
    # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_0&2/big_ML_biased_test/"
    # # cwd  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_init/"
    # cwd  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0&2_SL/"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    op_durations = {}
    

    # Data
    t0 = time.time()
    train_tagset_files_init, train_feature_matrix_init, train_label_matrix_init = tagsets_to_matrix(train_tags_init_path, cwd=cwd, train_flag=True, inference_flag=False, packages_select_set=train_packages_select_set, input_size=input_size)
    t1 = time.time()
    test_tagset_files_init, test_feature_matrix_init, test_label_matrix_init = tagsets_to_matrix(test_tags_path, cwd=cwd, train_flag=False, inference_flag=False, packages_select_set=test_packages_select_set, input_size=input_size)
    t2 = time.time()
    print(t1-t0)
    op_durations["tagsets_to_matrix-trainset"] = t1-t0
    op_durations["tagsets_to_matrix-trainset_xsize"] = train_feature_matrix_init.shape[0]
    op_durations["tagsets_to_matrix-trainset_ysize"] = train_feature_matrix_init.shape[1]
    op_durations["tagsets_to_matrix-testset"] = t2-t1

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
    x, y = [], []
    for idx in range(min(unique), max(unique)+1):
        x.append(idx)
        if idx in unique:
            y.extend(list(counts[np.where(unique == idx)]))
        else:
            y.append(0)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    y_normalized = [round(y_entry/sum(y)*100, 2) for y_entry in y]
    p = ax.bar(x,y_normalized)
    ax.bar_label(p, fontsize=18)
    # ax.set_xticklabels([str(idx) for idx in x])
    # ax.set_xticks(x)
    ax.set_xlim([0,5.5])
    ax.set_title("% of Tokens Occurring in Multiple Packages", fontsize=20)
    ax.set_ylabel("% of Tokens", fontsize=20)
    ax.set_xlabel("Number of Packages", fontsize=20)
    plt.savefig(cwd+'train_feature_init_used_count_freq_bar.pdf', format='pdf', dpi=50, bbox_inches='tight')
    plt.close(fig)
    gc.collect()

    # ######## Plot train label occurance as a B/W plot
    # fig, ax = plt.subplots(1, 1, figsize=(600, 10))
    # ax.imshow(train_label_matrix_init > 0, cmap='hot', interpolation="nearest")
    # plt.savefig(cwd+'train_label_matrix_init.pdf', format='pdf', dpi=50, bbox_inches='tight')
    # plt.close(fig)
    # gc.collect()

    # with open(cwd+'train_tagset_files_init.txt', 'w') as f:
    #     for line in train_tagset_files_init:
    #         f.write(f"{line}\n")


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

    # Init Training & Testing
    t0 = time.time()
    BOW_XGB_init = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=depth, learning_rate=0.1,silent=False, objective='binary:logistic', \
                      booster='gbtree', n_jobs=n_jobs, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                      subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, tree_method=tree_method, max_bin=max_bin)
                    #   subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1, tree_method=tree_method)
    BOW_XGB_init.fit(train_feature_matrix_init, train_label_matrix_init)
    t1 = time.time()
    pred_label_matrix_init = BOW_XGB_init.predict(test_feature_matrix_init)
    t2 = time.time()
    pred_label_prob_matrix_init = BOW_XGB_init.predict_proba(test_feature_matrix_init)
    t3 = time.time()
    results = one_hot_to_names(cwd+'index_label_mapping', pred_label_matrix_init)
    t4 = time.time()
    print(t1-t0, t2-t1, t3-t2)
    op_durations["BOW_XGB_init.fit"] = t1-t0
    op_durations["BOW_XGB_init.predict"] = t2-t1
    op_durations["BOW_XGB_init.predict_proba"] = t3-t2
    op_durations["one_hot_to_names"] = t4-t3

    # np.savetxt(cwd+'test_feature_matrix.out', test_feature_matrix, delimiter=',')

    # np.savetxt(cwd+'pred_label_matrix_init.out', pred_label_matrix_init, delimiter=',')
    # np.savetxt(cwd+'test_label_matrix_init.out', test_label_matrix_init, delimiter=',')
    # np.savetxt(cwd+'pred_label_prob_matrix_init.out', pred_label_prob_matrix_init, delimiter=',')
    # np.savetxt(cwd+'results.out', results, delimiter=',')
    with open(cwd+'results.out', 'w') as fp:
        labels = yaml.dump(results, fp)
    with open(cwd+"pred_d_dump", 'w') as writer:
        results_d = {}
        for k,v in results.items():
            results_d[int(k)] = v
        yaml.dump(results_d, writer)
    with open(cwd+'index_label_mapping', 'rb') as fp:
        labels = np.array(pickle.load(fp))
    print_metrics(cwd, 'metrics_init.out', test_label_matrix_init, pred_label_matrix_init, labels, op_durations)

    BOW_XGB_init.save_model(cwd+'model_init.json')


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

def run_pred(cwd, clf_path_l, test_tags_path, n_jobs=64, n_estimators=100, packages_select_set=set(), input_size=None, depth=1, tree_method="auto"):
    # # cwd = "/pipelines/component/cwd/"
    # cwd = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd/"
    # clf_path = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd/model_init.json"
    # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/inference_test/"
    Path(cwd).mkdir(parents=True, exist_ok=True)

    op_durations = {}
    label_matrix_list, pred_label_matrix_list, labels_list = [], [], []
    results = defaultdict(list)
    for clf_idx, clf_path in enumerate(clf_path_l):
        # op_durations = {}

        # # load from previous component
        # with open(test_tags_path, 'rb') as reader:
        #     tagsets_l = pickle.load(reader)
        t0 = time.time()
        # ########### convert from tag:count strings to encoding format
        tagset_files, feature_matrix, label_matrix = tagsets_to_matrix(test_tags_path, inference_flag=False, cwd=clf_path[:-15], packages_select_set=packages_select_set, input_size=input_size) # get rid of "model_init.json" in the clf_path.
        # # ########### load a previously converted encoding format data obj
        # with open(test_tags_path+"feature_matrix.obj","rb") as filehandler:
        #     feature_matrix = pickle.load(filehandler)
        # with open(test_tags_path+"label_matrix.obj","rb") as filehandler:
        #     label_matrix = pickle.load(filehandler)
        # with open(test_tags_path+"tagset_files.obj","rb") as filehandler:
        #     tagset_files = pickle.load(filehandler)
        # # ############################################
        BOW_XGB = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=depth, learning_rate=0.1,silent=False, objective='binary:logistic', \
                      booster='gbtree', n_jobs=n_jobs, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                      subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, tree_method=tree_method)
        BOW_XGB.load_model(clf_path)
        BOW_XGB.set_params(n_jobs=n_jobs)
        t1 = time.time()
        op_durations[clf_path+"\n BOW_XGB.load_model"] = t1-t0
        op_durations[clf_path+"\n feature_matrix_size_0"] = feature_matrix.shape[0]
        op_durations[clf_path+"\n feature_matrix_size_1"] = feature_matrix.shape[1]
        op_durations[clf_path+"\n label_matrix_size_0"] = label_matrix.shape[0]
        op_durations[clf_path+"\n label_matrix_size_1"] = label_matrix.shape[1]
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
        label_matrix_list.append(label_matrix)
        pred_label_matrix_list.append(pred_label_matrix)
        results = merge_preds(results, one_hot_to_names(clf_path[:-15]+'index_label_mapping', pred_label_matrix))
        with open(clf_path[:-15]+'index_label_mapping', 'rb') as fp:
            labels_list.append(np.array(pickle.load(fp)))
        t2 = time.time()
        op_durations[clf_path+"\n BOW_XGB.predict"] = t1-t0
        op_durations[clf_path+"\n one_hot_to_names"] = t2-t1

        # op_durations_glb[clf_path] = op_durations
        # results_d = {}
        # for k,v in results.items():
        #     results_d[int(k)] = v

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
    import itertools
    # # p = psutil.Process()
    # # p.cpu_affinity([0])
    # # ###################################
    # cwd  = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_8_6_rm_tmp_1_train/"
    # BOW_XGB = load_model(cwd+"model_init.json")
    # # booster = BOW_XGB.get_booster()
    # # tree_df = booster.trees_to_dataframe()
    # # print(BOW_XGB.get_num_boosting_rounds())

    # fig, ax = plt.subplots(figsize=(30, 30))
    # xgb.plot_tree(BOW_XGB, num_trees=1001, ax=ax)
    # plt.savefig(cwd+'btree1001.pdf', format='pdf', dpi=500, bbox_inches='tight')
    # fig, ax = plt.subplots(figsize=(30, 30))
    # xgb.plot_importance(BOW_XGB, ax=ax)
    # plt.savefig(cwd+'feature_importance.pdf', format='pdf', dpi=500, bbox_inches='tight')
    # print()


    ###################################
    # # run_init_train()
    packages_ll = {}
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
    # ============= data_3
    packages_l = ['certifi', 'numpy', 'packaging', 'aiobotocore', 'protobuf', 'jmespath', 'googleapis-common-protos', 'platformdirs', 'google-auth', 'werkzeug', 'pydantic', 'filelock', 'pyparsing', 'async-timeout', 'aiohttp', 'docutils', 'pyarrow', 'exceptiongroup', 'pluggy', 'lxml', 'requests-oauthlib', 'tqdm', 'pyasn1-modules', 'azure-core', 'decorator', 'pyopenssl', 'greenlet', 'importlib-resources', 'multidict', 'pygments', 'websocket-client', 'pymysql', 'distlib', 'coverage', 'aiosignal', 'et-xmlfile', 'openpyxl', 'chardet', 'google-cloud-core', 'google-cloud-storage', 'asn1crypto', 'tabulate', 'google-api-python-client', 'referencing', 'iniconfig', 'tomlkit', 'rpds-py', 'paramiko', 'gitpython', 'jsonschema-specifications', 'requests-toolbelt', 'pynacl', 'more-itertools', 'proto-plus', 'psycopg2-binary', 'itsdangerous', 'azure-storage-blob', 'msal', 'google-resumable-media', 'bcrypt', 'pathspec', 'tzlocal', 'anyio', 'grpcio-tools', 'google-cloud-bigquery', 'docker', 'cython', 'mdit-py-plugins', 'joblib', 'regex', 'mypy-extensions', 'smmap', 'gitdb', 'sagemaker', 'sqlparse', 'msgpack', 'wcwidth', 'google-auth-oauthlib', 'poetry-core', 'sniffio', 'py', 'pycryptodomex', 'pyrsistent', 'azure-common', 'future', 'dnspython', 'pexpect', 'ptyprocess', 'msrest', 'jaraco-classes', 'dill', 'portalocker', 'ruamel-yaml', 'markdown', 'snowflake-connector-python', 'py4j', 'tornado', 'keyring', 'google-crc32c', 'prompt-toolkit', 'markdown-it-py', 'tenacity', 'cloudpickle', 'httplib2', 'rich', 'alembic', 'gunicorn', 'tzdata', 'awswrangler', 'fonttools', 'azure-identity', 'threadpoolctl', 'msal-extensions', 'xmltodict', 'kiwisolver', 'pycodestyle', 'termcolor', 'python-dotenv', 'tb-nightly', 'scramp', 'backoff', 'uritemplate', 'toml', 'jedi', 'webencodings', 'cachecontrol', 'marshmallow', 'poetry-plugin-export', 'ipython', 'h11', 'mccabe', 'nest-asyncio', 'cycler', 'ply', 'sortedcontainers', 'pycryptodome', 'pg8000', 'google-auth-httplib2', 'trove-classifiers', 'oscrypto', 'traitlets', 'mako', 'pyodbc', 'pkgutil-resolve-name', 'pyzmq', 'prometheus-client', 'redshift-connector', 'isort', 'toolz', 'jeepney', 'httpcore', 'secretstorage', 'adal', 'pytest-cov', 'shellingham', 'babel', 'blinker', 'datadog', 'typing-inspect', 'black', 'pymongo', 'jsonpointer', 'jupyter-client', 'defusedxml', 'google-cloud-pubsub', 'argcomplete', 'httpx', 'tensorboard', 'pyflakes', 'jupyter-core', 'sentry-sdk', 'xlrd', 'flake8', 'poetry', 'cfn-lint', 'pkginfo', 'fastapi', 'nbconvert', 'mdurl', 'requests-aws4auth', 'parso', 'asynctest', 'contourpy', 'pydantic-core', 'python-json-logger', 'absl-py', 'jsonpath-ng', 'databricks-cli', 'python-utils', 'google-cloud-bigquery-storage', 'nbformat', 'pickleshare', 'backcall', 'fastjsonschema', 'notebook', 'progressbar2', 'astroid', 'aioitertools', 'mistune', 'starlette', 'rapidfuzz', 'matplotlib-inline', 'opensearch-py', 'appdirs', 'lazy-object-proxy', 'jupyter-server', 'tensorflow', 'ipykernel', 'pbr', 'pylint', 'transformers', 'arrow', 'h5py', 'kubernetes', 'build', 'jsonpatch', 'imageio', 'setuptools-scm', 'bleach', 'huggingface-hub', 'asgiref', 'annotated-types', 'websockets', 'html5lib', 'debugpy', 'cattrs', 'pyproject-hooks', 'entrypoints', 'grpc-google-iam-v1', 'uvicorn', 'mlflow', 'smart-open', 'oauth2client', 'altair', 'msrestazure', 'multiprocess', 'numba', 'tinycss2', 'dulwich', 'llvmlite', 'tensorflow-estimator', 'zope-interface', 'lockfile', 'elasticsearch', 'mock', 'google-pasta', 'flatbuffers', 'retry', 'aiofiles', 'google-cloud-secret-manager', 'pygithub', 'mypy', 'humanfriendly', 'requests-file', 'shapely', 'orjson', 'crashtest', 'great-expectations', 'aenum', 'pysocks', 'cleo', 'comm', 'httptools', 'gast', 'querystring-parser', 'nodeenv', 'nbclient', 'tensorboard-data-server', 'contextlib2', 'identify', 'xlsxwriter', 'cached-property', 'azure-storage-file-datalake', 'croniter', 'tox', 'deepdiff', 'tokenizers', 'django', 'notebook-shim', 'send2trash', 'mysql-connector-python', 'ipywidgets', 'configparser', 'pendulum', 'execnet', 'jupyterlab-server', 'widgetsnbextension', 'text-unidecode', 'rfc3339-validator', 'overrides', 'pre-commit', 'typer', 'keras', 'json5', 'semver', 'watchdog', 'hvac', 'responses', 'torch', 'jupyterlab', 'pytzdata', 'aws-sam-translator', 'snowflake-sqlalchemy', 'python-slugify', 'cfgv', 'pipenv', 'asttokens', 'argon2-cffi', 'installer', 'dataclasses', 'sphinx', 'jupyterlab-widgets', 'executing', 'gremlinpython', 'distro', 'typeguard', 'azure-mgmt-core', 'selenium', 'jupyter-events', 'pytest-xdist', 'click-plugins', 'stack-data', 'pytest-mock', 'azure-storage-common', 'confluent-kafka', 'slack-sdk', 'pure-eval', 'opt-einsum', 'rfc3986', 'xgboost', 'tblib', 'dataclasses-json', 'opentelemetry-sdk', 'apache-airflow', 'uri-template', 'fastavro', 'tensorflow-serving-api', 'ipython-genutils', 'sentencepiece', 'futures', 'tensorflow-io-gcs-filesystem', 'sympy', 'unidecode', 'xxhash', 'safetensors', 'db-dtypes', 'pandocfilters', 'prettytable', 'patsy', 'opentelemetry-api', 'retrying', 'docopt', 'azure-mgmt-resource', 'mpmath', 'gcsfs', 'async-lru', 'jupyterlab-pygments', 'astunparse', 'setproctitle', 'terminado', 'libclang', 'pytest-runner', 'thrift', 'jsonpickle', 'semantic-version', 'ordered-set', 'azure-keyvault-secrets', 'pymssql', 'faker', 'pysftp', 'webcolors', 'argon2-cffi-bindings', 'jupyter-lsp', 'typing', 'rfc3986-validator', 'zeep', 'inflection', 'antlr4-python3-runtime', 'sphinxcontrib-serializinghtml', 'azure-datalake-store', 'graphviz', 'boto', 'fqdn', 'isoduration', 'jupyter-server-terminals', 'deprecation', 'moto', 'snowballstemmer', 'openai', 'opentelemetry-proto', 'distributed', 'azure-graphrbac', 'typed-ast', 'sphinxcontrib-htmlhelp', 'sphinxcontrib-applehelp', 'sphinxcontrib-devhelp', 'sphinxcontrib-qthelp', 'opencensus', 'ujson', 'opencensus-context', 'aioconsole', 'pathos', 'libcst', 'parsedatetime', 'stevedore', 'python-gnupg', 'google-cloud-firestore', 'pyproj', 'pandas-gbq', 'pox', 'trio', 'ppft', 'gspread', 'applicationinsights', 'numexpr', 'gevent', 'zope-event', 'kombu', 'shap', 'argparse', 'opentelemetry-exporter-otlp-proto-http', 'trio-websocket', 'google-cloud-appengine-logging', 'email-validator', 'structlog', 'loguru', 'watchtower', 'pyathena', 'torchvision', 'azure-mgmt-keyvault', 'azure-mgmt-storage', 'simple-salesforce', 'checkov', 'coloredlogs', 'apache-beam', 'tensorboard-plugin-wit', 'gsutil', 'kafka-python', 'mypy-boto3-rds', 'celery', 'pathlib2', 'pycrypto', 'wandb', 'colorlog', 'enum34', 'pybind11', 'tldextract', 'prometheus-flask-exporter', 'opentelemetry-semantic-conventions', 'types-urllib3', 'azure-cosmos', 'azure-eventhub', 'djangorestframework', 'opencensus-ext-azure', 'docstring-parser', 'lz4', 'pydata-google-auth', 'pywavelets', 'lightgbm', 'datetime', 'ecdsa', 'pyhcl', 'uamqp', 'cligj', 'google-cloud-resource-manager', 'slicer', 'fire', 'makefun', 'python-jose', 'azure-mgmt-containerregistry', 'imagesize', 'google-cloud-logging', 'keras-preprocessing', 'unittest-xml-reporting', 'alabaster', 'flask-cors', 'schema', 'hpack', 'nvidia-cudnn-cu11', 'partd', 'delta-spark', 'nvidia-cublas-cu11', 'wsproto', 'amqp', 'hypothesis', 'pytest-asyncio', 'python-http-client', 'validators', 'h2', 'azure-mgmt-authorization', 'databricks-sql-connector', 'sshtunnel', 'hyperframe', 'spacy', 'unicodecsv', 'brotli', 'fiona', 'locket', 'apache-airflow-providers-common-sql', 'holidays']
    packages_ll["data_3"]=packages_l

    for dataset in ["data_0"]:
        packages_l = packages_ll[dataset]
        for n_jobs in [8]:
            for n_models in [32,16,8,1]:
                for n_estimators in [100]:
                    for depth in [1]:
                        for tree_method in["exact"]: # "exact","approx","hist"
                            for max_bin in [1]:
                                for input_size in [None]: # [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
                                    package_subset, step = [], int(len(packages_l)/n_models)
                                    for i in range(0, len(packages_l), step):
                                        package_subset.append(set(packages_l[i:i+step]))

                                    for i, train_subset in enumerate(package_subset):
                                        test_subset = set()
                                        for package_names in itertools.permutations(train_subset, 2):
                                            test_subset.add("-".join(package_names))
                                        # train_tags_path = "/home/ubuntu/data_0/big_train"+dataset+"/big_train/"
                                        # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/big_ML_biased_test/"
                                        train_tags_path = "/home/ubuntu/data_0/big_train/"
                                        test_tags_path = "/home/ubuntu/data_0/big_ML_biased_test/"
                                        cwd  ="/home/ubuntu/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par/"
                                        # cwd  ="/home/ubuntu/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_"+str(n_models)+"_"+str(i)+"_train_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"rawinput_sampling1_"+str(tree_method)+"treemethod_doublesamples/"
                                        # cwd  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_"+str(n_models)+"_"+str(i)+"_train_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"randomint10000000_sampling1_"+str(tree_method)+"treemethod_doublesamples/"
                                        run_init_train(train_tags_path, test_tags_path, cwd, n_jobs=n_jobs, n_estimators=n_estimators, train_packages_select_set=train_subset, test_packages_select_set=test_subset, input_size=input_size, depth=depth, tree_method=tree_method)
                                        # break




    ###################################
    # run_iter_train()


    # # ###################################
    # # run_pred()
    # for dataset in ["data_0"]:
    #     for n_jobs in [1]:
    #         for clf_njobs in [8]:
    #             for n_models in [8]:
    #                 for n_estimators in [100]:
    #                     for depth in [1]:
    #                         for tree_method in["exact"]: # "exact","approx","hist"
    #                             for max_bin in [1]:
    #                                 for input_size in [None]:
    #                                     clf_path = []
    #                                     for i in range(n_models):
    #                                         clf_path.append("/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(clf_njobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par/model_init.json")
    #                                     cwd = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par/"
    #                                     test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/big_ML_biased_test/"
    #                     #    run_init_train(train_tags_path, test_tags_path, cwd, n_jobs=n_jobs, n_estimators=n_estimators, train_packages_select_set=train_subset, test_packages_select_set=test_subset, input_size=input_size, depth=depth, tree_method=tree_method)
    #                                     run_pred(cwd, clf_path, test_tags_path, n_jobs=n_jobs, n_estimators=n_estimators, input_size=input_size, depth=depth, tree_method=tree_method)


    ###################################
    # verify the init trees are still tehere.
    #   Preconfigure significantly large feature and label spaces
    #       Both VW and XGBoost might have this problem
    #       NN seems to be an easier way for fast incremental training, i.e., simply replace the output and input layer, until your model is not strong enough.
    # evaluation steps
    # compare VW and XGBoost
    # TF-IDF
    # NIW: 