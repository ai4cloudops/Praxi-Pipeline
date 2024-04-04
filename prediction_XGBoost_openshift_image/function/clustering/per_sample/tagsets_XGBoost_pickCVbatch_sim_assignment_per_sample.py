import os, pickle, time, gc
import yaml, json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.datasets import make_multilabel_classification
import sklearn.metrics as metrics
from sklearn.preprocessing import Normalizer
import scipy
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

def map_tagfilesl(tags_path, tag_files, cwd, inference_flag, freq=100, tokens_filter_set=set()):
    all_tags_set, all_label_set = set(), set()
    tags_by_instance_l, labels_by_instance_l = [], []
    tagset_files = []
    data_instance_d_l = [read_tokens(tags_path, tag_file, cwd, inference_flag, freq=freq, tokens_filter_set=tokens_filter_set) for tag_file in tag_files]
    # data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
    for data_instance_d in data_instance_d_l:
        if len(data_instance_d) ==4:
                tagset_files.append(data_instance_d['tag_file'])
                all_tags_set.update(data_instance_d['local_all_tags_set'])
                tags_by_instance_l.append(data_instance_d['instance_feature_tags_d'])
                all_label_set.update(data_instance_d['labels'])
                labels_by_instance_l.append(data_instance_d['labels'])
    return {"tagset_files": tagset_files, "all_tags_set": all_tags_set, "tags_by_instance_l":tags_by_instance_l ,"all_label_set":all_label_set, "labels_by_instance_l":labels_by_instance_l}

def read_tokens(tags_path, tag_file, cwd, inference_flag, freq=100, tokens_filter_set=set()):
    ret = {}
    ret["tag_file"] = tag_file
    try:
        # if(tag_file[-3:] == 'tag') and (tag_file[:-3].rsplit('-', 1)[0] in packages_select_set or packages_select_set == set()):
        with open(tags_path + tag_file, 'rb') as tf:
            # print(tag_file)
            # tagset_files.append(tag_file)
            # local_all_tags_set = set()
            # instance_feature_tags_d = defaultdict(int)
            tagset = yaml.load(tf, Loader=yaml.Loader)
            # tagset = json.load(tf)   

            # feature 
            local_all_tags_set = set(tagset['tags'].keys())
            instance_feature_tags_d = tagset['tags']
            # filtered_tags_l = list()
            # for k,v in tagset['tags'].items():
            # # for tag_vs_count in tagset['tags']:
            # #     k,v = tag_vs_count.split(":")
            #     # if k not in tokens_filter_set:
            #     #     local_all_tags_set.add(k)
            #     #     instance_feature_tags_d[k] += int(v)
            #     # else:
            #     #     filtered_tags_l.append(k)
            #     if k not in tokens_filter_set or tokens_filter_set[k] < freq:
            #         local_all_tags_set.add(k)
            #         instance_feature_tags_d[k] += int(v)
            #     else:
            #         filtered_tags_l.append(k)
            # if local_all_tags_set == set():
            #     logger = build_logger(tag_file, cwd+"logs/")
            #     logger.info('%s', tag_file+" has empty tags after filtering: "+str(filtered_tags_l))
            #     return ret
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

def tagsets_to_matrix(tags_path, tag_files_l = None, index_tag_mapping_path=None, tag_index_mapping_path=None, index_label_mapping_path=None, label_index_mapping_path=None, cwd="/home/cc/Praxi-study/Praxi-Pipeline/prediction_base_image/model_testing_scripts/cwd/", train_flag=False, inference_flag=True, iter_flag=False, packages_select_set=set(), tokens_filter_set=set(), input_size=None, compact_factor=1, freq=100, all_tags_set=None,all_label_set=None,tags_by_instance_l=None,labels_by_instance_l=None,tagset_files=None):
    if index_tag_mapping_path == None:
        index_tag_mapping_path=cwd+'index_tag_mapping'
        tag_index_mapping_path=cwd+'tag_index_mapping'
        index_label_mapping_path=cwd+'index_label_mapping'
        label_index_mapping_path=cwd+'label_index_mapping'

        index_tag_mapping_iter_path=cwd+"index_tag_mapping_iter"
        tag_index_mapping_iter_path=cwd+"tag_index_mapping_iter"
        index_label_mapping_iter_path=cwd+"index_label_mapping_iter"
        label_index_mapping_iter_path=cwd+"label_index_mapping_iter"
    
    if all_tags_set == None:
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
        # pool = mp.Pool(processes=mp.cpu_count())
        pool = mp.Pool(processes=32)
        data_instance_d_l = [pool.apply_async(map_tagfilesl, args=(tags_path, tag_files_l, cwd, inference_flag, freq), kwds={"tokens_filter_set": tokens_filter_set}) for tag_files_l in tqdm(tag_files_l_of_l)]
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
        for data_instance_d in data_instance_d_l:
            if len(data_instance_d) == 5:
                    tagset_files.extend(data_instance_d['tagset_files'])
                    all_tags_set.update(data_instance_d['all_tags_set'])
                    tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
                    all_label_set.update(data_instance_d['all_label_set'])
                    labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
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



    # # #############
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
        instance_row_list.append(scipy.sparse.csr_matrix(instance_row))
        # instance_row_list.append(instance_row)
    # instance_row_list.extend(instance_row_list)
    feature_matrix = scipy.sparse.vstack(instance_row_list)
    # feature_matrix = np.vstack(instance_row_list)
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
            # instance_row_list.append(scipy.sparse.csr_matrix(instance_row))
            instance_row_list.append(instance_row)
            # label_matrix = np.vstack([label_matrix, instance_row])
        # label_matrix = np.delete(label_matrix, (0), axis=0)
        # instance_row_list.extend(instance_row_list)
        # label_matrix = scipy.sparse.vstack(instance_row_list)
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
    # print(len(y_true), len(y_true[0]), len(y_pred), len(labels))
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

def run_init_train(train_tags_init_path, test_tags_path, cwd, train_tags_init_l=None, test_tags_l=None, n_jobs=64, n_estimators=100, train_packages_select_set=set(), highlight_label_set=set(), tokens_filter_set=set(), test_packages_select_set=set(), test_batch_count=1, input_size=None, compact_factor=1, depth=1, tree_method="auto", max_bin=6, freq=100):
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
    train_tagset_files_init, train_feature_matrix_init, train_label_matrix_init = tagsets_to_matrix(train_tags_init_path, tag_files_l=train_tags_init_l, cwd=cwd, train_flag=True, inference_flag=False, packages_select_set=train_packages_select_set, tokens_filter_set=tokens_filter_set, input_size=input_size, compact_factor=compact_factor, freq=freq)
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

    # ######## Plot average train feature usage per label as a B/W plot
    # # train_feature_init_used_count = (train_feature_matrix_init > 0).sum(axis=0)
    # # train_label_init_used_count = (train_label_matrix_init > 0).sum(axis=1)
    # # label_count_d = defaultdict(int)
    # # for line in train_tagset_files_init:
    # #     label_count_d["-".join(line.split("-")[:-1])] += 1
    # train_feature_init_used_count_list = []
    # idxs_yx = np.nonzero(train_label_matrix_init)
    # label_row_idx = np.array([])
    # col_idx_prev = -1
    # row_idx_prev = -1
    # for entry_idx, (row_idx, col_idx) in enumerate(zip(idxs_yx[0],idxs_yx[1])):
    #     if col_idx_prev != col_idx:
    #         if col_idx_prev != -1:
    #             train_feature_init_used_count_list.append(train_feature_matrix_init[list(range(row_idx_prev, row_idx)), :].mean(axis=0))
    #         col_idx_prev = col_idx
    #         row_idx_prev = row_idx
    #         label_row_idx = np.append(label_row_idx, [row_idx])
    #     if entry_idx == len(idxs_yx[0])-1 and col_idx_prev != -1:
    #         train_feature_init_used_count_list.append(train_feature_matrix_init[list(range(row_idx_prev, row_idx+1)), :].mean(axis=0))
    # train_feature_init_used_count = np.vstack(train_feature_init_used_count_list)
    # # fig, ax = plt.subplots(1, 1, figsize=(600, 10))
    # # ax.imshow(train_feature_init_used_count > 0, cmap='hot', interpolation="nearest")
    # # plt.savefig(cwd+'train_feature_init_used_count.pdf', format='pdf', dpi=50, bbox_inches='tight')
    # # plt.close(fig)
    # # gc.collect()
    # # fig, ax = plt.subplots(1, 1, figsize=(600, 10))
    # # ax.bar(list(range(train_feature_matrix_init.shape[1])), (train_feature_init_used_count > 0).sum(axis=0))
    # # plt.savefig(cwd+'train_feature_init_used_count_bar.pdf', format='pdf', dpi=50, bbox_inches='tight')
    # # plt.close(fig)
    # # gc.collect()
    # unique, counts = np.unique((train_feature_init_used_count > 0).sum(axis=0), return_counts=True)
    # feature_occur, feature_occurence_count = [], []
    # for idx in range(min(unique), max(unique)+1):
    #     feature_occur.append(idx)
    #     if idx in unique:
    #         feature_occurence_count.extend(list(counts[np.where(unique == idx)]))
    #     else:
    #         feature_occurence_count.append(0)



    # if len(highlight_label_set) != 0:
    #     with open(cwd+'label_index_mapping', 'rb') as fp:
    #         label_index_mapping = pickle.load(fp)
    #     highlight_label_train_feature_init_used_count_list = []
    #     for label in highlight_label_set:
    #         highlight_label_train_feature_init_used_count_list.append(train_feature_matrix_init[np.where(train_label_matrix_init[:,label_index_mapping[label]] == 1)].mean(axis=0))
    #     highlight_label_train_feature_init_used_count = np.vstack(highlight_label_train_feature_init_used_count_list)
    #     highlight_label_feature_occur, highlight_label_feature_occurence_count = np.unique((highlight_label_train_feature_init_used_count > 0).sum(axis=0), return_counts=True)



    

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

    # # Data Distribution Summary
    # if len(highlight_label_set) != 0:
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    #     ax.tick_params(axis='both', which='major', labelsize=20)
    #     ax.tick_params(axis='both', which='minor', labelsize=18)
    #     zipped = list(zip(highlight_label_feature_occur, highlight_label_feature_occurence_count))
    #     zipped.sort(key=lambda x: x[1],reverse=True)
    #     highlight_label_feature_occur, highlight_label_feature_occurence_count = zip(*zipped)
    #     highlight_label_feature_occurence_count_normalized = [round(highlight_label_feature_occurence_count_entry/sum(highlight_label_feature_occurence_count)*100, 2) for highlight_label_feature_occurence_count_entry in highlight_label_feature_occurence_count]
    #     p = ax.bar([idx for idx in range(len(highlight_label_feature_occurence_count_normalized))],highlight_label_feature_occurence_count_normalized)
    #     ax.bar_label(p, fontsize=18)
    #     ax.set_xticklabels([str(occur) for occur in highlight_label_feature_occur])
    #     ax.set_xticks([idx for idx in range(len(highlight_label_feature_occurence_count_normalized))])
    #     ax.set_xlim([-0.5,5.5])
    #     ax.set_title("% of Tokens Occurring in Multiple Packages", fontsize=20)
    #     ax.set_ylabel("% of Tokens", fontsize=20)
    #     ax.set_xlabel("Number of Packages", fontsize=20)
    #     plt.savefig(cwd+'highlight_label_train_feature_init_used_count_freq_bar.pdf', format='pdf', dpi=50, bbox_inches='tight')
    #     plt.close(fig)
    #     gc.collect()

    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=18)
    # zipped = list(zip(feature_occur, feature_occurence_count))
    # zipped.sort(key=lambda x: x[1],reverse=True)
    # feature_occur, feature_occurence_count = zip(*zipped)
    # feature_occurence_count_normalized = [round(feature_occurence_count_entry/sum(feature_occurence_count)*100, 2) for feature_occurence_count_entry in feature_occurence_count]
    # p = ax.bar([idx for idx in range(len(feature_occurence_count_normalized))],feature_occurence_count_normalized)
    # ax.bar_label(p, fontsize=18)
    # ax.set_xticklabels([str(occur) for occur in feature_occur])
    # ax.set_xticks([idx for idx in range(len(feature_occurence_count_normalized))])
    # ax.set_xlim([-0.5,5.5])
    # ax.set_title("% of Tokens Occurring in Multiple Packages", fontsize=20)
    # ax.set_ylabel("% of Tokens", fontsize=20)
    # ax.set_xlabel("Number of Packages", fontsize=20)
    # plt.savefig(cwd+'train_feature_init_used_count_freq_bar.pdf', format='pdf', dpi=50, bbox_inches='tight')
    # plt.close(fig)
    # gc.collect()





    # Testing Epochs
    # with open(cwd+'index_label_mapping', 'rb') as fp:
    #     labels = np.array(pickle.load(fp))
    label_matrix_list, pred_label_matrix_list = [], []
    if test_tags_l == None:
        test_tags_l = [tag_file for tag_file in os.listdir(test_tags_path) if tag_file[-3:] == 'tag']
    step = len(test_tags_l)//test_batch_count
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
    # Load tag:count in mapping format 
    with open(test_tags_path+"all_tags_set.obj","rb") as filehandler:
        all_tags_set = pickle.load(filehandler)
    with open(test_tags_path+"all_label_set.obj","rb") as filehandler:
        all_label_set = pickle.load(filehandler)
    with open(test_tags_path+"tags_by_instance_l.obj","rb") as filehandler:
        tags_by_instance_l = pickle.load(filehandler)
    with open(test_tags_path+"labels_by_instance_l.obj","rb") as filehandler:
        labels_by_instance_l = pickle.load(filehandler)
    with open(test_tags_path+"tagset_files.obj","rb") as filehandler:
        tagset_files = pickle.load(filehandler)

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
            tagset_files, feature_matrix, label_matrix = tagsets_to_matrix(test_tags_path, tag_files_l = tag_files_l[batch_first_idx:batch_first_idx+step], inference_flag=False, cwd=clf_path[:-15], packages_select_set=packages_select_set, input_size=input_size, compact_factor=compact_factor, all_tags_set=all_tags_set,all_label_set=all_label_set,tags_by_instance_l=tags_by_instance_l,labels_by_instance_l=labels_by_instance_l,tagset_files=tagset_files) # get rid of "model_init.json" in the clf_path.
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
            # with open(clf_path[:-15]+'index_label_mapping', 'rb') as fp:
            #     labels = np.array(pickle.load(fp))
            # print_metrics(cwd, 'metrics_pred_'+str(batch_first_idx)+'.out', label_matrix, pred_label_matrix, labels, op_durations)
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
    package_ver_dd = {}
    n_samples_d = {}
    test_portion_d = {}
    tokenshares_filter_set_d = {}
    tokennoises_filter_set_d = {}
    clustering_d = {}
    # ============= data_0
    n_samples_d["data_0"]=25
    test_portion_d["data_0"]=0.2

    # ============= data_3
    n_samples_d["data_3"]=21
    test_portion_d["data_3"]=0.2

    # ============= data_4
    n_samples_d["data_4"]=4
    test_portion_d["data_4"]=0.25
    clustering_d[0.9] = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4_clustering/name_groups.json"


    # ###################### choose CV batch ######################
    for test_sample_batch_idx in [0]:   # this is the initial idx of a test batch, i.e., if test batch size is 4*0.25 = 1, `test_sample_batch_idx`` means pick the 0-index element as the test batch.
        # #############################################################

        for dataset in ["data_4"]:
            n_samples = n_samples_d[dataset]
            test_portion = test_portion_d[dataset]
            for (with_filter, freq) in [(False, 100)]:
                if with_filter:
                    # Consider a set with tokens to filter
                    with open(f"/home/cc/Praxi-study/Praxi-Pipeline/data/{dataset}/filters/tagsets_SL_tagnames_reoccurentcount_d", 'rb') as tf:
                        tokens_filter_set = yaml.load(tf, Loader=yaml.Loader)
                else:
                    tokens_filter_set = set()
                for n_jobs in [32]:
                    # for n_models, test_batch_count in zip([25, 10, 1],[1,1,1,1,1]):
                    for n_models, sim_thr, test_batch_count in zip([3000],[0.9],[1,1,1,1,1]):
                        for n_estimators in [100]:
                            for depth in [1]:
                                for tree_method in["exact"]: # "exact","approx","hist"
                                    for max_bin in [1]:
                                        for input_size, dim_compact_factor in zip([None],[1,1,1,1,1]):
                                            with open(clustering_d[sim_thr], 'r') as openfile:
                                                # Reading from json file
                                                name_groups = json.load(openfile)
                                                n_models = min(len(name_groups["train_name_groups"]), n_models)
                                            random_instance = random.Random(4)
                                            for shuffle_idx in range(3):

                                                zipped = list(zip(name_groups["train_name_groups"], name_groups["test_name_groups"]))

                                                # Shuffle the list of groups to randomize the order
                                                shuffled_groups = random.sample(zipped, len(zipped))
                                                
                                                # Calculate the number of groups we aim to combine into one to reach the desired number of groups
                                                target_group_size = len(zipped) // n_models
                                                if len(zipped) % n_models:  # Adjust if we can't divide groups evenly
                                                    target_group_size += 1

                                                regrouped = []
                                                for i in range(0, len(shuffled_groups), target_group_size):
                                                    # Combine the next 'target_group_size' groups into one
                                                    combined_group = []
                                                    for group in shuffled_groups[i:i + target_group_size]:
                                                        combined_group.extend(group)
                                                    regrouped.append(combined_group)

                                                # train_subsets_l, test_subsets_l = zip(*regrouped)

                                                # cross-validation for samples
                                                for i, train_subset, test_subset in enumerate(regrouped):
                                                    train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/tagsets_SL/"
                                                    test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/tagsets_SL/" # Cross Validation: testing a portion of the SL dataset

                                                    train_tag_files_l = list(train_subset)
                                                    test_tag_files_l = list(test_subset)
                                                    cwd  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_simassignment/"
                                                    run_init_train(train_tags_path, test_tags_path, cwd, train_tags_init_l=train_tag_files_l, test_tags_l=test_tag_files_l, n_jobs=n_jobs, n_estimators=n_estimators, tokens_filter_set=tokens_filter_set, test_batch_count=test_batch_count, input_size=input_size, compact_factor=dim_compact_factor, depth=depth, tree_method=tree_method, freq=freq)
                                                    # break


        ###################################
        # run_iter_train()


        # ###################################
        # run_pred()
        # Testing the ML dataset
        for dataset in ["data_4"]:
            n_samples = n_samples_d[dataset]
            for (with_filter, freq) in [(False, 100)]:
                for n_jobs in [32]:
                    for clf_njobs in [32]:
                        for n_models, test_batch_count in zip([25, 10, 1],[1,1,1,1]): # zip([50,25,20,15,10,5,1],[1,1,1,1,1,1,8]):
                            for n_estimators in [100]:
                                for depth in [1]:
                                    for tree_method in["exact"]: # "exact","approx","hist"
                                        for max_bin in [1]:
                                            for input_size, dim_compact_factor in zip([None],[1,1,1,1,1]):
                                                for shuffle_idx in range(3):

                                                    clf_path = []
                                                    for i in range(n_models):
                                                        clf_pathname = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(clf_njobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_noverpak/model_init.json"
                                                        if os.path.isfile(clf_pathname):
                                                            clf_path.append(clf_pathname)
                                                        else:
                                                            print("clf is missing!")
                                                            sys.exit(-1)
                                                    cwd = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_noverpak/"
                                                    # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/big_ML_biased_test/"
                                                    test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/tagsets_ML_2/"
                                    #    run_init_train(train_tags_path, test_tags_path, cwd, n_jobs=n_jobs, n_estimators=n_estimators, train_packages_select_set=train_subset, test_packages_select_set=test_subset, input_size=input_size, depth=depth, tree_method=tree_method)
                                                    run_pred(cwd, clf_path, test_tags_path, n_jobs=n_jobs, n_estimators=n_estimators, test_batch_count=test_batch_count, input_size=input_size, compact_factor=dim_compact_factor, depth=depth, tree_method=tree_method)











    # # ########## some debugging scripts.
    # # d = defaultdict(list)
    # # for test_tag_file in train_tag_files_l:
    # #     d[test_tag_file[:-4].rsplit("-",1)[0]].append(test_tag_file)
    # # for k,v in d.items():
    # #     # print(k,len(v))
    # #     if len(v) != 5:
    # #         print("!!!!!")
    # #         break

    # # aenum_v3_1_13.0.tag


    # import json
    
    # # Opening JSON file
    # with open('/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/dockerfiles_SL/inventory.json', 'r') as openfile:
    #     # Reading from json file
    #     json_object = json.load(openfile)
    #     print(list(set([x[0] for x in json_object])))
    #     print(list(set([x[0].split("==")[0] for x in json_object])))
                                                    



    # import yaml
    # with open(f"/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/filters/tagsets_SL_tagnames_reoccurentcount_d", 'rb') as tf:
    #     tokens_filter_set = yaml.load(tf, Loader=yaml.Loader)
    # tokens_filter_set_100lite = {}
    # for k, v in tokens_filter_set.items():
    #     if v >= 100:
    #         tokens_filter_set_100lite[k] = v
    # with open(f"/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/filters/tagsets_SL_tagnames_reoccurentcount_100lite_d", 'w') as tf:
    #     yaml.dump(tokens_filter_set_100lite, tf)