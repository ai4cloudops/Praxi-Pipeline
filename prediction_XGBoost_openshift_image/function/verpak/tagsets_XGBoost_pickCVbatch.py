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
            local_all_tags_set = set()
            instance_feature_tags_d = defaultdict(int)
            tagset = yaml.load(tf, Loader=yaml.Loader)
            # tagset = json.load(tf)   

            # feature 
            # local_all_tags_set = set(tagset['tags'].keys())
            # instance_feature_tags_d = tagset['tags']
            filtered_tags_l = list()
            for k,v in tagset['tags'].items():
            # for tag_vs_count in tagset['tags']:
            #     k,v = tag_vs_count.split(":")
                # if k not in tokens_filter_set:
                #     local_all_tags_set.add(k)
                #     instance_feature_tags_d[k] += int(v)
                # else:
                #     filtered_tags_l.append(k)
                if k not in tokens_filter_set or tokens_filter_set[k] < freq:
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

def tagsets_to_matrix(tags_path, tag_files_l = None, index_tag_mapping_path=None, tag_index_mapping_path=None, index_label_mapping_path=None, label_index_mapping_path=None, cwd="/home/cc/Praxi-study/Praxi-Pipeline/prediction_base_image/model_testing_scripts/cwd/", train_flag=False, inference_flag=True, iter_flag=False, packages_select_set=set(), tokens_filter_set=set(), input_size=None, compact_factor=1, freq=100, all_tags_set=None,all_label_set=None,tags_by_instance_l=None,labels_by_instance_l=None,tagset_files=None):
    op_durations = {}
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
            t_read_tag_files_l_0 = time.time()
            tag_files_l = [tag_file for tag_file in os.listdir(tags_path) if (tag_file[-3:] == 'tag') and (tag_file[:-4].rsplit('-', 1)[0] in packages_select_set or packages_select_set == set())]
            op_durations["t_read_tag_files_l"] = time.time()-t_read_tag_files_l_0
        # return 
        t_prepare_tag_files_l_of_l_0 = time.time()
        # tag_files_l_of_l, step = [], len(tag_files_l)//mp.cpu_count()+1
        tag_files_l_of_l, step = [], max(len(tag_files_l)//32,1)
        for i in range(0, len(tag_files_l), step):
            tag_files_l_of_l.append(tag_files_l[i:i+step])
        op_durations["t_prepare_tag_files_l_of_l"] = time.time()-t_prepare_tag_files_l_of_l_0
        t_load_tag_files_l_0 = time.time()
        # pool = mp.Pool(processes=mp.cpu_count())
        pool = mp.Pool(processes=32)
        data_instance_d_l = [pool.apply_async(map_tagfilesl, args=(tags_path, tag_files_l, cwd, inference_flag, freq), kwds={"tokens_filter_set": tokens_filter_set}) for tag_files_l in tqdm(tag_files_l_of_l)]
        data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
        pool.close()
        pool.join()
        op_durations["t_load_tag_files_l"] = time.time()-t_load_tag_files_l_0
        t_dup_data_instance_d_l_0 = time.time()
        for data_instance_d in data_instance_d_l:
            if len(data_instance_d) == 5:
                    tagset_files.extend(data_instance_d['tagset_files'])
                    all_tags_set.update(data_instance_d['all_tags_set'])
                    tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
                    all_label_set.update(data_instance_d['all_label_set'])
                    labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
        # for data_instance_d in data_instance_d_l:
        #     if len(data_instance_d) == 5:
        #             tagset_files.extend(data_instance_d['tagset_files'])
        #             all_tags_set.update(data_instance_d['all_tags_set'])
        #             tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
        #             all_label_set.update(data_instance_d['all_label_set'])
        #             labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
        # for data_instance_d in data_instance_d_l:
        #     if len(data_instance_d) == 5:
        #             tagset_files.extend(data_instance_d['tagset_files'])
        #             all_tags_set.update(data_instance_d['all_tags_set'])
        #             tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
        #             all_label_set.update(data_instance_d['all_label_set'])
        #             labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
        # for data_instance_d in data_instance_d_l:
        #     if len(data_instance_d) == 5:
        #             tagset_files.extend(data_instance_d['tagset_files'])
        #             all_tags_set.update(data_instance_d['all_tags_set'])
        #             tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
        #             all_label_set.update(data_instance_d['all_label_set'])
        #             labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
        # for data_instance_d in data_instance_d_l:
        #     if len(data_instance_d) == 5:
        #             tagset_files.extend(data_instance_d['tagset_files'])
        #             all_tags_set.update(data_instance_d['all_tags_set'])
        #             tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
        #             all_label_set.update(data_instance_d['all_label_set'])
        #             labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
        # for data_instance_d in data_instance_d_l:
        #     if len(data_instance_d) == 5:
        #             tagset_files.extend(data_instance_d['tagset_files'])
        #             all_tags_set.update(data_instance_d['all_tags_set'])
        #             tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
        #             all_label_set.update(data_instance_d['all_label_set'])
        #             labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
        # for data_instance_d in data_instance_d_l:
        #     if len(data_instance_d) == 5:
        #             tagset_files.extend(data_instance_d['tagset_files'])
        #             all_tags_set.update(data_instance_d['all_tags_set'])
        #             tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
        #             all_label_set.update(data_instance_d['all_label_set'])
        #             labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
        # for data_instance_d in data_instance_d_l:
        #     if len(data_instance_d) == 5:
        #             tagset_files.extend(data_instance_d['tagset_files'])
        #             all_tags_set.update(data_instance_d['all_tags_set'])
        #             tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
        #             all_label_set.update(data_instance_d['all_label_set'])
        #             labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
        op_durations["t_dup_data_instance_d_l"] = time.time()-t_dup_data_instance_d_l_0
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
        t_sorting_data_instance_d_l_0 = time.time()
        if not inference_flag:
            zipped = list(zip(tagset_files, tags_by_instance_l, labels_by_instance_l))
            zipped.sort(key=lambda x: x[0])
            tagset_files, tags_by_instance_l, labels_by_instance_l = zip(*zipped)
        else:
            zipped = list(zip(tagset_files, tags_by_instance_l))
            zipped.sort(key=lambda x: x[0])
            tagset_files, tags_by_instance_l = zip(*zipped)
        op_durations["t_sorting_data_instance_d_l"] = time.time()-t_sorting_data_instance_d_l_0



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
    t_generate_feat_mapping_0 = time.time()
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
    op_durations["t_generate_feat_mapping"] = time.time()-t_generate_feat_mapping_0
    ## Generate Feature Matrix
    op_durations["t_generate_feat_mat_np_zeros"] = 0
    t_generate_feat_mat_0 = time.time()
    instance_row_list = []
    for instance_tags_d in tags_by_instance_l:
        if input_size == None:
            input_size = len(all_tags_l)//compact_factor
        t_generate_feat_mat_np_zeros_0 = time.time()
        instance_row = np.zeros(input_size)
        op_durations["t_generate_feat_mat_np_zeros"] += time.time()-t_generate_feat_mat_np_zeros_0
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
    t_generate_feat_mat_vtack_0 = time.time()
    feature_matrix = scipy.sparse.vstack(instance_row_list)
    # feature_matrix = np.vstack(instance_row_list)
    op_durations["t_generate_feat_mat:vstack"] = time.time()-t_generate_feat_mat_vtack_0
    del instance_row_list
    # with open(cwd+'removed_tags_l', 'wb') as fp:
    #     pickle.dump(removed_tags_l, fp)
    # with open(cwd+'removed_tags_l.txt', 'w') as f:
    #     for line in removed_tags_l:
    #         f.write(f"{line}\n")
    op_durations["t_generate_feat_mat"] = time.time()-t_generate_feat_mat_0
    


    # Label Matrix Generation
    label_matrix = np.array([])
    if not inference_flag:
        removed_label_l = []
        ## Handling Mapping
        t_generate_label_mapping_0 = time.time()
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
        op_durations["t_generate_label_mapping"] = time.time()-t_generate_label_mapping_0
        ## Handling Label Matrix
        t_generate_label_mat_0 = time.time()
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
        op_durations["t_generate_label_mat"] = time.time()-t_generate_label_mat_0
    
    return tagset_files, feature_matrix, label_matrix, op_durations
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
        # file_header += ("\n {:-^55}\n".format("DURATION REPORT") + "\n".join(["{}:{:.3f}".format(k, v) for k, v in op_durations.items()]))
        with open(f"{cwd}measurement_metrics.yaml", 'w') as writer:
            yaml.dump(op_durations, writer)
    
    
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
    print(cwd)
    t0 = time.time()
    train_tagset_files_init, train_feature_matrix_init, train_label_matrix_init, op_durations_encoder = tagsets_to_matrix(train_tags_init_path, tag_files_l=train_tags_init_l, cwd=cwd, train_flag=True, inference_flag=False, packages_select_set=train_packages_select_set, tokens_filter_set=tokens_filter_set, input_size=input_size, compact_factor=compact_factor, freq=freq)
    # print(process.memory_info())
    t1 = time.time()
    print(t1-t0)
    op_durations["tagsets_to_matrix-trainset"] = t1-t0
    op_durations["tagsets_to_matrix-trainset_details"] = op_durations_encoder
    op_durations["tagsets_to_matrix-trainset_xsize"] = train_feature_matrix_init.shape[0]
    op_durations["tagsets_to_matrix-trainset_ysize"] = train_feature_matrix_init.shape[1]
    print("============================Encoder========================")
    print(op_durations)
    print("===========================================================")

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
        test_tagset_files_init, test_feature_matrix_init, test_label_matrix_init, op_durations_encoder = tagsets_to_matrix(test_tags_path, tag_files_l=test_tags_l[batch_first_idx:batch_first_idx+step], cwd=cwd, train_flag=False, inference_flag=False, packages_select_set=test_packages_select_set, input_size=input_size, compact_factor=compact_factor)
        # print(process.memory_info())
        t1 = time.time()
        op_durations["tagsets_to_matrix-testset_"+str(batch_first_idx)] = t1-t0
        op_durations["tagsets_to_matrix-testset_"+str(batch_first_idx)+"_details"] = op_durations_encoder
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

def run_pred(cwd, clf_path_l, test_tags_path, tag_files_l=None, flag_load_obj=True, n_jobs=64, n_estimators=100, packages_select_set=set(), test_batch_count=1, input_size=None, compact_factor=1, depth=1, tree_method="auto"):
    op_durations = {}
    # # cwd = "/pipelines/component/cwd/"
    # cwd = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd/"
    # clf_path = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd/model_init.json"
    # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/inference_test/"
    Path(cwd).mkdir(parents=True, exist_ok=True)
    inference_flag = False

    if flag_load_obj:
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
    else:
        all_tags_set, all_label_set = set(), set()
        tags_by_instance_l, labels_by_instance_l = [], []
        tagset_files = []
        if tag_files_l == None:
            t_read_test_tag_files_l_0 = time.time()
            tag_files_l = [tag_file for tag_file in os.listdir(test_tags_path) if tag_file[-3:] == 'tag']
            op_durations["t_read_tag_files_l"] = time.time()-t_read_test_tag_files_l_0
        t_prepare_test_tag_files_l_of_l_0 = time.time()
        # tag_files_l_of_l, step = [], len(tag_files_l)//mp.cpu_count()+1
        tag_files_l_of_l, step = [], len(tag_files_l)//32+1
        for i in range(0, len(tag_files_l), step):
            tag_files_l_of_l.append(tag_files_l[i:i+step])
        op_durations["t_prepare_test_tag_files_l_of_l"] = time.time()-t_prepare_test_tag_files_l_of_l_0
        t_load_test_tag_files_l_0 = time.time()
        # pool = mp.Pool(processes=mp.cpu_count())
        pool = mp.Pool(processes=32)
        data_instance_d_l = [pool.apply_async(map_tagfilesl, args=(test_tags_path, tag_files_l, cwd, inference_flag, freq), kwds={"tokens_filter_set": tokens_filter_set}) for tag_files_l in tqdm(tag_files_l_of_l)]
        data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
        pool.close()
        pool.join()
        op_durations["t_load_test_tag_files_l"] = time.time()-t_load_test_tag_files_l_0
        t_dup_test_data_instance_d_l_0 = time.time()
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
        op_durations["t_dup_test_data_instance_d_l"] = time.time()-t_dup_test_data_instance_d_l_0
        # Sorting instances
        t_sorting_test_data_instance_d_l_0 = time.time()
        if not inference_flag:
            zipped = list(zip(tagset_files, tags_by_instance_l, labels_by_instance_l))
            zipped.sort(key=lambda x: x[0])
            tagset_files, tags_by_instance_l, labels_by_instance_l = zip(*zipped)
        else:
            zipped = list(zip(tagset_files, tags_by_instance_l))
            zipped.sort(key=lambda x: x[0])
            tagset_files, tags_by_instance_l = zip(*zipped)
        op_durations["t_sorting_test_data_instance_d_l"] = time.time()-t_sorting_test_data_instance_d_l_0

    
    label_matrix_list, pred_label_matrix_list, labels_list = [], [], []
    results = defaultdict(list)
    for clf_idx, clf_path in enumerate(clf_path_l):
        print(cwd)
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
        # tag_files_l = [tag_file for tag_file in os.listdir(test_tags_path) if tag_file[-3:] == 'tag']
        step = len(tag_files_l)//test_batch_count+1
        for batch_first_idx in range(0, len(tag_files_l), step):
            # op_durations = {}

            # # load from previous component
            # with open(test_tags_path, 'rb') as reader:
            #     tagsets_l = pickle.load(reader)
            t0 = time.time()
            # ########### convert from tag:count strings to encoding format
            tagset_files_used, feature_matrix, label_matrix, op_durations_encoder = tagsets_to_matrix(test_tags_path, tag_files_l = tag_files_l[batch_first_idx:batch_first_idx+step], inference_flag=inference_flag, cwd=clf_path[:-15], packages_select_set=packages_select_set, input_size=input_size, compact_factor=compact_factor, all_tags_set=all_tags_set,all_label_set=all_label_set,tags_by_instance_l=tags_by_instance_l,labels_by_instance_l=labels_by_instance_l,tagset_files=tagset_files) # get rid of "model_init.json" in the clf_path.
            # # ########### load a previously converted encoding format data obj
            # with open(test_tags_path+"feature_matrix.obj","rb") as filehandler:
            #     feature_matrix = pickle.load(filehandler)
            # with open(test_tags_path+"label_matrix.obj","rb") as filehandler:
            #     label_matrix = pickle.load(filehandler)
            # with open(test_tags_path+"tagset_files_used.obj","rb") as filehandler:
            #     tagset_files_used = pickle.load(filehandler)
            # # ############################################
            t1 = time.time()
            op_durations[clf_path+"\n tagsets_to_matrix-testset"+str(batch_first_idx)+"/"+str(test_batch_count)] = t1-t0
            op_durations[clf_path+"\n tagsets_to_matrix-testset"+str(batch_first_idx)+"/"+str(test_batch_count)+"_details"] = op_durations_encoder
            op_durations[clf_path+"\n feature_matrix_size_0_"+str(batch_first_idx)+"/"+str(test_batch_count)] = feature_matrix.shape[0]
            op_durations[clf_path+"\n feature_matrix_size_1_"+str(batch_first_idx)+"/"+str(test_batch_count)] = feature_matrix.shape[1]
            op_durations[clf_path+"\n label_matrix_size_0_"+str(batch_first_idx)+"/"+str(test_batch_count)] = label_matrix.shape[0]
            op_durations[clf_path+"\n label_matrix_size_1_"+str(batch_first_idx)+"/"+str(test_batch_count)] = label_matrix.shape[1]
            # op_durations[clf_path+"\n tagset_files_used"] = tagset_files_used
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
    # with open(test_tags_path+"tagset_files_used.obj","wb") as filehandler:
    #     pickle.dump(tagset_files_used,filehandler)
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
    # packages_l = ['pypdf2', 'jupyterlab-pygments', 'lazy-object-proxy', 'sqlalchemy', 'feedparser', 'google-cloud-videointelligence', 'sentence-transformers', 'oscrypto', 'python-dateutil', 'pyspnego', 'python-levenshtein', 'evaluate', 'avro-python3', 'ppft', 'validators', 'prometheus-flask-exporter', 'freezegun', 'tldextract', 'streamlit', 'deprecated', 'arrow', 'authlib', 'tinycss2', 'pycryptodomex', 'contourpy', 'partd', 'parse-type', 'simple-salesforce', 'nose', 'tensorflow-text', 'google-cloud-kms', 'httpx', 'cfgv', 'urllib3', 'referencing', 'pycryptodome', 'antlr4-python3-runtime', 'azure-mgmt-botservice', 'pytest-xdist', 'cx-oracle', 'flask-sqlalchemy', 'mpmath', 'scramp', 'pycparser', 'aws-requests-auth', 'adal', 'defusedxml', 'pylint', 'enum34', 'azure-kusto-ingest', 'python-daemon', 'pygments', 'pyproject-api', 'moto', 'sqlalchemy-utils', 'azure-mgmt-security', 'future', 'nltk', 'jupyter-events', 'gensim', 'rich-argparse', 'keras-preprocessing', 'apache-airflow-providers-common-sql', 'shap', 'keras-applications', 'azure-storage-common', 'frozendict', 'wcwidth', 'openai', 'google-cloud-dlp', 'lark', 'pathy', 'imagesize', 'pluggy', 'smmap', 'python-dotenv', 'google-cloud-audit-log', 'srsly', 'nvidia-cudnn-cu12', 'azure-mgmt-datafactory', 'trio-websocket', 'click', 'types-redis', 'pyhive', 'ndg-httpsclient', 'packaging', 'dbt-extractor', 'pox', 'azure-graphrbac', 'azure-keyvault', 'xyzservices', 'azure-mgmt-redhatopenshift', 'types-setuptools', 'tifffile', 'sqlalchemy-jsonfield', 'timm', 'botocore', 'aiohttp', 'uvloop', 'nvidia-nccl-cu12', 'fonttools', 'netaddr', 'ecdsa', 'azure-mgmt-containerservice', 'google-cloud-language', 'aioitertools', 'azure-mgmt-iotcentral', 'click-repl', 'spacy-legacy', 'django-extensions', 'spark-nlp', 'httpcore', 'awswrangler', 'pooch', 'apache-airflow-providers-http', 'certifi', 'aniso8601', 'azure-mgmt-applicationinsights', 'flask-appbuilder', 'configobj', 'mmh3', 'pymsteams', 'azure-common', 'dm-tree', 'azure-mgmt-batchai', 'pexpect', 'aws-lambda-powertools', 'pyzmq', 'azure-mgmt-reservations', 'pyee', 'google-re2', 'annotated-types', 'html5lib', 'hatchling', 'installer', 'twisted', 'reportlab', 'azure-mgmt-cdn', 'omegaconf', 'pycares', 'azure-mgmt-search', 'flask-babel', 'traitlets', 'distributed', 'msal', 'fastavro', 'boltons', 'dask', 'mergedeep', 'nvidia-nvjitlink-cu12', 'bitarray', 'astroid', 'mypy-boto3-appflow', 'types-requests', 'torchvision', 'cloudpathlib', 'soupsieve', 'pgpy', 'greenlet', 'catalogue', 'azure-mgmt-cosmosdb', 'texttable', 'termcolor', 'kfp-server-api', 'opentelemetry-exporter-otlp-proto-grpc', 'jupyterlab', 'korean-lunar-calendar', 'bcrypt', 'scikit-image', 'qtconsole', 'astunparse', 'pydot', 'geoip2', 'cssselect', 'qtpy', 'azure-mgmt-rdbms', 'sortedcontainers', 'looker-sdk', 'apache-airflow', 'pydash', 'tokenizers', 'astor', 'torch', 'google-api-core', 'parso', 'azure-cosmos', 'filelock', 'evidently', 'cymem', 'ldap3', 'levenshtein', 'py-cpuinfo', 'cramjam', 'rapidfuzz', 'fire', 'toml', 'checkov', 'more-itertools', 'gremlinpython', 'langchain', 'nvidia-cufft-cu12', 'matplotlib', 'kfp-pipeline-spec', 'azure-mgmt-imagebuilder', 'imbalanced-learn', 'mkdocs-material', 'apscheduler', 'responses', 'typing-extensions', 'time-machine', 'pyelftools', 'pyjwt', 'smdebug-rulesconfig', 'contextlib2', 'azure-mgmt-resource', 'flask', 'exceptiongroup', 'websockets', 'terminado', 'pyflakes', 'funcsigs', 'nvidia-curand-cu12', 'locket', 'cycler', 'openapi-spec-validator', 'tensorboard-plugin-wit', 'parameterized', 'comm', 'apache-airflow-providers-ftp', 'setuptools', 'google-cloud-bigquery-storage', 'slack-sdk', 'docopt', 'pycountry', 'grpcio-tools', 'widgetsnbextension', 'jaraco-classes', 'mock', 'notebook-shim', 'opentelemetry-proto', 'uc-micro-py', 'faker', 'azure-mgmt-kusto', 'ijson', 'websocket-client', 'gcsfs', 'python-jsonpath', 'python-jose', 'cython', 'secretstorage', 'opentelemetry-api', 'webdriver-manager', 'asyncio', 'types-awscrt', 'tensorflow', 'elastic-transport', 'requests-ntlm', 'importlib-resources', 'hologram', 'azure-mgmt-appconfiguration', 'opentelemetry-exporter-otlp-proto-common', 'junit-xml', 'natsort', 'starlette', 'azure-mgmt-keyvault', 'protobuf3-to-dict', 'polars', 'pysocks', 'python-utils', 'limits', 'user-agents', 'azure-mgmt-trafficmanager', 'jupyter-server', 'poetry-core', 'multiprocess', 'typeguard', 'pickleshare', 'simplejson', 'requests-file', 'msrest', 'pymeeus', 'pyasn1', 'tensorflow-io', 'sphinxcontrib-qthelp', 'ipywidgets', 'fastparquet', 'dbt-core', 'setproctitle', 'incremental', 'google-cloud-resource-manager', 'pyrsistent', 'pyspark', 'marshmallow-sqlalchemy', 'google-cloud-vision', 'gast', 'requests-aws4auth', 'agate', 'overrides', 'argcomplete', 'scp', 'libclang', 'rpds-py', 'tox', 'mccabe', 'uri-template', 'fasteners', 'dpath', 'send2trash', 'twine', 'apache-beam', 'xlsxwriter', 'cachelib', 'boto3-stubs', 'pathos', 'grpcio', 'snowflake-sqlalchemy', 'pytzdata', 'google-cloud-pubsub', 'xxhash', 'rich', 'datadog', 'scandir', 'flask-jwt-extended', 'sh', 'azure-mgmt-network', 'monotonic', 'semver', 'jsonpointer', 'flask-cors', 'cron-descriptor', 'pysftp', 'pyotp', 'prompt-toolkit', 'confluent-kafka', 'azure-mgmt-storage', 'gevent', 'flask-wtf', 'dulwich', 'azure-mgmt-advisor', 'retry', 'azure-mgmt-marketplaceordering', 'google-cloud-datacatalog', 'sshtunnel', 'py4j', 'pyarrow', 'pypandoc', 'databricks-sdk', 'commonmark', 'cachetools', 'retrying', 'charset-normalizer', 'pywavelets', 'nvidia-cudnn-cu11', 'watchtower', 'h2', 'azure-mgmt-msi', 'hypothesis', 'types-protobuf', 's3fs', 'thrift', 'prometheus-client', 'mypy-boto3-s3', 'pytest-metadata', 'ddsketch', 'markdown-it-py', 'azure-keyvault-administration', 'wtforms', 'aiodns', 'kr8s', 'azure-multiapi-storage', 'azure-mgmt-eventgrid', 'office365-rest-python-client', 'unicodecsv', 'slicer', 'progressbar2', 's3transfer', 'tiktoken', 'murmurhash', 'azure-mgmt-containerregistry', 'typing-inspect', 'redis', 'psycopg', 'anyio', 'dvclive', 'openpyxl', 'bytecode', 'azure-mgmt-monitor', 'ptyprocess', 'ruamel-yaml', 'pbr', 'google', 'opensearch-py', 'cligj', 'nh3', 'ray', 'kombu', 'datetime', 'toolz', 'semantic-version', 'hiredis', 'rfc3986', 'jsonpatch', 'qrcode', 'marshmallow', 'smart-open', 'django-filter', 'docstring-parser', 'build', 'configupdater', 'datadog-api-client', 'async-generator', 'json5', 'idna', 'factory-boy', 'cleo', 'pynacl', 'scikit-learn', 'xarray', 'mistune', 'cached-property', 'imageio', 'proto-plus', 'bandit', 'azure-mgmt-compute', 'readme-renderer', 'azure-identity', 'azure-data-tables', 'evergreen-py', 'azure-eventgrid', 'huggingface-hub', 'stringcase', 'execnet', 'tensorboard', 'azure-mgmt-recoveryservices', 'dacite', 'seaborn', 'newrelic', 'robotframework-seleniumlibrary', 'zstandard', 'tensorflow-hub', 'gunicorn', 'marshmallow-enum', 'geopandas', 'psycopg2-binary', 'mashumaro', 'flatbuffers', 'resolvelib', 'onnxruntime', 'atomicwrites', 'ipaddress', 'nvidia-cublas-cu12', 'types-pyyaml', 'ansible', 'sphinx', 'geographiclib', 'jeepney', 'statsmodels', 'zope-interface', 'google-cloud-logging', 'xgboost', 'pydata-google-auth', 'pytest-html', 'ml-dtypes', 'sendgrid', 'google-auth', 'phonenumbers', 'ordered-set', 'hpack', 'shapely', 'configparser', 'sagemaker', 'ninja', 'makefun', 'azure-batch', 'iso8601', 'apache-airflow-providers-databricks', 'msrestazure', 'flake8', 'click-plugins', 'jpype1', 'tqdm', 'fastapi', 'ratelimit', 'tenacity', 'pathspec', 'cfn-lint', 'aws-psycopg2', 'coverage', 'types-s3transfer', 'six', 'alembic', 'requests-oauthlib', 'great-expectations', 'pg8000', 'pika', 'pymongo', 'configargparse', 'safetensors', 'cmake', 'jaydebeapi', 'pymysql', 'aiosignal', 'coloredlogs', 'docker', 'apache-airflow-providers-cncf-kubernetes', 'azure-cli-core', 'psutil', 'pytest-asyncio', 'tabulate', 'snowballstemmer', 'cloudpickle', 'selenium', 'plotly', 'isort', 'applicationinsights', 'jupyter-core', 'wheel', 'ftfy', 'kiwisolver', 'zeep', 'prettytable', 'asttokens', 'asyncache', 'nvidia-cuda-runtime-cu12', 'pandas', 'prison', 'loguru', 'ddtrace', 'webencodings', 'typing', 'jax', 'fsspec', 'httplib2', 'pyaml', 'ujson', 'distrax', 'cachecontrol', 'notebook', 'passlib', 'llvmlite', 'twilio', 'cog', 'nvidia-cuda-nvrtc-cu12', 'grpcio-status', 'jedi', 'webcolors', 'typer', 'gradio', 'google-cloud-compute', 'pytorch-lightning', 'sentry-sdk', 'chardet', 'aws-xray-sdk', 'pathlib', 'shellingham', 'sphinxcontrib-htmlhelp', 'mypy', 'aws-sam-translator', 'kafka-python', 'python-slugify', 'structlog', 'types-pytz', 'knack', 'portalocker', 'threadpoolctl', 'grpcio-health-checking', 'alabaster', 'apache-airflow-providers-ssh', 'jsonschema', 'markupsafe', 'text-unidecode', 'mypy-boto3-redshift-data', 'argparse', 'py', 'apache-airflow-providers-sqlite', 'watchfiles', 'grpc-google-iam-v1', 'google-cloud-build', 'pathlib2', 'cookiecutter', 'stack-data', 'envier', 'ruff', 'inflection', 'tzlocal', 'voluptuous', 'wrapt', 'decorator', 'confection', 'constructs', 'spacy-loggers', 'pyproj', 'azure-mgmt-batch', 'rsa', 'google-pasta', 'ultralytics', 'geopy', 'azure-synapse-artifacts', 'pyserial', 'azure-mgmt-consumption', 'jupyterlab-server', 'cerberus', 'pure-eval', 'regex', 'gitpython', 'google-cloud-datastore', 'slackclient', 'apache-airflow-providers-amazon', 'azure-mgmt-eventhub', 'croniter', 'wsproto', 'pandocfilters', 'mako', 'networkx', 'pydantic-core', 'fqdn', 'jmespath', 'sentencepiece', 'jira', 'google-cloud-bigtable', 'oldest-supported-numpy', 'azure-eventhub', 'hyperlink', 'db-dtypes', 'delta-spark', 'tensorflow-estimator', 'azure-synapse-accesscontrol', 'altair', 'thinc', 'numpy', 'google-ads', 'azure-mgmt-loganalytics', 'azure-keyvault-keys', 'platformdirs', 'googleapis-common-protos', 'scipy', 'aenum', 'pyhcl', 'nbclassic', 'pytest-cov', 'msal-extensions', 'azure-cli-telemetry', 'azure-synapse-spark', 'pre-commit', 'azure-storage-file-datalake', 'typed-ast', 'unidecode', 'azure-mgmt-servicebus', 'pybind11', 'google-cloud-spanner', 'executing', 'azure-mgmt-netapp', 'djangorestframework', 'pyparsing', 'python-multipart', 'google-cloud-bigquery', 'querystring-parser', 'google-cloud-storage', 'cryptography', 'pytest-mock', 'azure-mgmt-datamigration', 'azure-keyvault-certificates', 'libcst', 'celery', 'pycodestyle', 'black', 'azure-mgmt-recoveryservicesbackup', 'nest-asyncio', 'humanize', 'itsdangerous', 'sphinxcontrib-applehelp', 'fuzzywuzzy', 'setuptools-rust', 'boto3', 'cffi', 'tensorflow-metadata', 'lightgbm', 'jinja2', 'azure-mgmt-cognitiveservices', 'python-gnupg', 'fabric', 'aiofiles', 'elasticsearch', 'dill', 'pyyaml', 'bracex', 'gsutil', 'nodeenv', 'jsondiff', 'pygithub', 'beautifulsoup4', 'h5py', 'azure-storage-file-share', 'jupyterlab-widgets', 'deepdiff', 'google-resumable-media', 'pendulum', 'pillow', 'dvc-render', 'sphinxcontrib-serializinghtml', 'tensorflow-datasets', 'ipython', 'kornia', 'schema', 'hvac', 'mysql-connector-python', 'docutils', 'identify', 'azure-mgmt-media', 'nvidia-cuda-cupti-cu12', 'elasticsearch-dsl', 'azure-datalake-store', 'azure-mgmt-policyinsights', 'python-docx', 'jsonpickle', 'pytest-forked', 'vine', 'lxml', 'leather', 'distro', 'h11', 'aiobotocore', 'opencensus', 'google-cloud-monitoring', 'azure-mgmt-devtestlabs', 'gql', 'stevedore', 'azure-mgmt-apimanagement', 'jupyter-client', 'azure-mgmt-authorization', 'poetry', 'dataclasses', 'statsd', 'bokeh', 'sphinxcontrib-devhelp', 'unittest-xml-reporting', 'virtualenv', 'langdetect', 'hdfs', 'cdk-nag', 'python-gitlab', 'apache-airflow-providers-imap', 'azure-appconfiguration', 'babel', 'awscli', 'async-lru', 'rdflib', 'opencv-python-headless', 'uvicorn', 'shortuuid', 'graphql-core', 'langsmith', 'pydantic', 'google-cloud-tasks', 'azure-mgmt-core', 'python-box', 'starkbank-ecdsa', 'spacy', 'gitdb', 'logbook', 'pypdf', 'ipykernel', 'oauthlib', 'azure-mgmt-iothubprovisioningservices', 'orbax-checkpoint', 'lightning-utilities', 'diskcache', 'paramiko', 'tensorboard-data-server', 'sqlparse', 'google-cloud-core', 'poetry-plugin-export', 'databricks-sql-connector', 'google-cloud-container', 'invoke', 'patsy', 'nbclient', 'xlwt', 'django-cors-headers', 'cmdstanpy', 'pytest-runner', 'tzdata', 'opencensus-ext-azure', 'tornado', 'google-crc32c', 'trove-classifiers', 'bleach', 'kubernetes', 'jaxlib', 'markdown', 'opentelemetry-exporter-otlp-proto-http', 'hyperframe', 'apispec', 'uritemplate', 'yapf', 'appdirs', 'transformers', 'setuptools-scm', 'lockfile', 'kfp', 'uamqp', 'email-validator', 'marshmallow-dataclass', 'dbt-postgres', 'absl-py', 'javaproperties', 'pytimeparse', 'trio', 'sphinx-rtd-theme', 'yarl', 'google-cloud-secret-manager', 'dataclasses-json', 'parse', 'azure-mgmt-dns', 'frozenlist', 'google-cloud-firestore', 'tomlkit', 'google-cloud-appengine-logging', 'orjson', 'pyopenssl', 'flit-core', 'amqp', 'apache-airflow-providers-slack', 'parsedatetime', 'pytest-timeout', 'accelerate', 'torchmetrics', 'sympy', 'databricks-cli', 'numba', 'azure-mgmt-iothub', 'nvidia-cusparse-cu12', 'preshed', 'azure-mgmt-containerinstance', 'pymssql', 'jsonlines', 'inflect', 'mypy-extensions', 'jupyter-server-terminals', 'ansible-core', 'flask-caching', 'azure-mgmt-datalake-analytics', 'click-man', 'magicattr', 'deprecation', 'service-identity', 'async-timeout', 'azure-mgmt-datalake-store', 'entrypoints', 'outcome', 'asynctest', 'tomli', 'opentelemetry-exporter-otlp', 'colorama', 'nbformat', 'nvidia-nvtx-cu12', 'matplotlib-inline', 'attrs', 'opencv-python', 'keyring', 'ruamel-yaml-clib', 'pytest', 'pep517', 'azure-storage-queue', 'google-auth-oauthlib', 'types-urllib3', 'pyasn1-modules', 'flask-login', 'redshift-connector', 'wasabi', 'keras', 'emoji', 'python-magic', 'tensorflow-serving-api', 'numexpr', 'python-json-logger', 'tblib', 'connexion', 'cinemagoer', 'blis', 'backoff', 'promise', 'azure-mgmt-synapse', 'pyathena', 'imdbpy', 'pyperclip', 'tensorflow-probability', 'wandb', 'azure-keyvault-secrets', 'asgiref', 'pydub', 'autopep8', 'langcodes', 'requests', 'werkzeug', 'types-python-dateutil', 'jsonschema-specifications', 'gradio-client', 'distlib', 'databricks-api', 'fastjsonschema', 'google-cloud-dataproc', 'msgpack', 'fiona', 'google-auth-httplib2', 'maxminddb', 'azure-core', 'azure-mgmt-sqlvirtualmachine', 'billiard', 'firebase-admin', 'einops', 'linkify-it-py', 'ua-parser', 'iniconfig', 'python-editor', 'tensorflow-io-gcs-filesystem', 'debugpy', 'pandas-gbq', 'importlib-metadata', 'nvidia-cusolver-cu12', 'user-agent', 'yamllint', 'opt-einsum', 'pytz', 'zipp', 'convertdate', 'snowflake-connector-python', 'zope-event', 'rfc3339-validator', 'mlflow', 'humanfriendly', 'ec2-metadata', 'multidict', 'click-didyoumean', 'jupyter-console', 'cattrs', 'mdit-py-plugins', 'datasets', 'azure-mgmt-maps', 'protobuf', 'oauth2client', 'dnspython', 'pyodbc', 'azure-mgmt-servicefabric', 'mypy-boto3-rds', 'telethon', 'pytest-rerunfailures', 'onnx', 'jsonpath-ng', 'azure-storage-blob', 'django', 'colorlog', 'requests-mock', 'graphviz', 'holidays', 'google-api-python-client', 'azure-mgmt-signalr', 'h3', 'flax', 'pydeequ', 'azure-servicebus', 'sniffio', 'joblib', 'ply', 'botocore-stubs', 'apache-airflow-providers-snowflake', 'universal-pathlib', 'watchdog', 'azure-cli', 'editables', 'python-http-client', 'db-contrib-tool', 'argon2-cffi', 'gspread', 'azure-mgmt-web', 'nbconvert', 'lz4', 'azure-kusto-data', 'xlrd', 'avro', 'chex', 'docker-pycreds', 'azure-mgmt-sql', 'boto', 'opentelemetry-sdk', 'addict', 'flask-limiter', 'dateparser', 'jupyter-lsp', 'tableauserverclient', 'asn1crypto', 'xmltodict', 'tensorstore', 'azure-mgmt-hdinsight', 'google-cloud-aiplatform', 'azure-mgmt-redis', 'requests-toolbelt', 'httptools', 'blinker', 'flask-session']
    packages_l = ['office365-rest-python-client==2.5.6', 'great-expectations==0.18.10', 'levenshtein==0.25.0', 'gast==0.5.4', 'azure-mgmt-sql==2.1.0', 'mergedeep==1.3.3', 'azure-keyvault-administration==4.3.0', 'xlrd==1.2.0', 'twisted==23.10.0', 'azure-mgmt-search==9.1.0', 'keras==3.0.5', 'blis==0.7.11', 'newrelic==9.7.0', 'pyperclip==1.8.0', 'jupyter-client==8.4.0', 'pyzmq==25.1.1', 'vine==1.3.0', 'einops==0.6.1', 'colorlog==6.8.0', 'freezegun==1.3.1', 'cycler==0.12.1', 'pyhive==0.7.0', 'nbconvert==7.16.2', 'bracex==2.2.1', 'pytest-cov==4.1.0', 'azure-mgmt-consumption==10.0.0', 'statsd==3.3.0', 'mypy-boto3-appflow==1.29.0', 'h5py==3.8.0', 'psutil==5.9.8', 'google-cloud-spanner==3.43.0', 'azure-mgmt-cognitiveservices==13.4.0', 'matplotlib==3.8.1', 'pluggy==1.3.0', 'opentelemetry-proto==1.23.0', 'tomli==2.0.0', 'py4j==0.10.9.7', 'requests-ntlm==1.1.0', 'pyhcl==0.4.5', 'argparse==1.3.0', 'geoip2==4.6.0', 'docker-pycreds==0.3.0', 'datasets==2.17.0', 'itsdangerous==2.1.0', 'cmdstanpy==1.2.0', 'nbclassic==1.0.0', 'marshmallow-enum==1.4.1', 'markdown-it-py==3.0.0', 'antlr4-python3-runtime==4.13.1', 'readme-renderer==41.0', 'cfn-lint==0.85.3', 'maxminddb==2.5.2', 'azure-mgmt-batch==17.0.0', 'libclang==15.0.6.1', 'ninja==1.11.1.1', 'hatchling==1.21.1', 'azure-mgmt-recoveryservicesbackup==7.0.0', 'pyelftools==0.30', 'simplejson==3.18.4', 'prometheus-flask-exporter==0.23.0', 'sphinxcontrib-devhelp==1.0.6', 'looker-sdk==24.2.1', 'starkbank-ecdsa==2.1.0', 'feedparser==6.0.10', 'yamllint==1.34.0', 'azure-synapse-artifacts==0.17.0', 'sphinxcontrib-serializinghtml==1.1.9', 'opencv-python-headless==4.8.0.76', 'opensearch-py==2.4.2', 'backoff==2.1.2', 'slicer==0.0.7', 'docutils==0.20.1', 'humanize==4.7.0', 'scipy==1.12.0', 'attrs==23.2.0', 'apache-beam==2.53.0', 'pyopenssl==24.0.0', 'nvidia-curand-cu12==10.3.4.107', 'gunicorn==21.2.0', 'apache-airflow-providers-amazon==8.19.0', 'onnxruntime==1.16.3', 'monotonic==1.4', 'azure-mgmt-servicefabric==2.0.0', 'requests==2.29.0', 'azure-mgmt-compute==30.4.0', 'pyrsistent==0.20.0', 'rich==13.7.1', 'stevedore==5.0.0', 's3transfer==0.10.0', 'pycparser==2.20', 'tableauserverclient==0.29', 'isort==5.13.1', 'tensorstore==0.1.56', 'apispec==6.6.0', 'scramp==1.4.2', 'fastparquet==2023.10.1', 'azure-mgmt-trafficmanager==1.0.0', 'click==8.1.7', 'dataclasses-json==0.6.3', 's3fs==2024.2.0', 'jupyterlab-server==2.25.3', 'notebook-shim==0.2.4', 'astor==0.7.1', 'azure-mgmt-kusto==3.3.0', 'matplotlib-inline==0.1.3', 'omegaconf==2.3.0', 'atomicwrites==1.4.1', 'ec2-metadata==2.13.0', 'tensorflow-metadata==1.14.0', 'nest-asyncio==1.5.9', 'widgetsnbextension==4.0.8', 'pycountry==22.3.5', 'azure-mgmt-authorization==3.0.0', 'trio==0.24.0', 'grpcio==1.60.1', 'cffi==1.16.0', 'google-cloud-vision==3.7.2', 'cramjam==2.8.0', 'ujson==5.9.0', 'fuzzywuzzy==0.17.0', 'avro==1.11.1', 'newrelic==9.7.1', 'gcsfs==2023.12.1', 'pexpect==4.7.0', 'fastparquet==2024.2.0', 'time-machine==2.13.0', 'azure-eventgrid==4.17.0', 'webencodings==0.4', 'aenum==3.1.13', 'pygments==2.17.2', 'ppft==1.7.6.7', 'bokeh==3.3.3', 'great-expectations==0.18.11', 'prompt-toolkit==3.0.41', 'zope-event==4.5.0', 'frozendict==2.3.10', 'db-contrib-tool==0.6.12', 'grpcio-status==1.62.0', 'regex==2023.10.3', 'cdk-nag==2.28.65', 'azure-multiapi-storage==1.0.0', 'google-cloud-bigquery==3.18.0', 'azure-kusto-ingest==4.3.1', 'confluent-kafka==2.3.0', 'sphinxcontrib-qthelp==1.0.6', 'ipykernel==6.29.1', 'cloudpickle==3.0.0', 'parse==1.19.1', 'checkov==3.2.38', 'ddsketch==2.0.3', 'async-lru==2.0.2', 'setuptools-rust==1.8.1', 'nvidia-cublas-cu12==12.3.2.9', 'azure-cli-telemetry==1.0.7', 'tzdata==2023.3', 'ecdsa==0.18.0', 'asn1crypto==1.5.0', 'pyjwt==2.6.0', 'iso8601==2.0.0', 'ecdsa==0.16.1', 'distlib==0.3.8', 'google-auth-httplib2==0.1.1', 'django==4.2.9', 'kfp-pipeline-spec==0.2.2', 'numexpr==2.8.8', 'tomlkit==0.12.2', 'azure-mgmt-maps==2.0.0', 'sentencepiece==0.1.98', 'google-cloud-storage==2.13.0', 'cached-property==1.5.1', 'libcst==1.1.0', 'promise==2.3', 'mccabe==0.6.1', 'sendgrid==6.11.0', 'azure-mgmt-security==4.0.0', 'flask-limiter==3.4.1', 'apache-airflow-providers-imap==3.3.2', 'jupyterlab-widgets==3.0.8', 'nbclassic==0.5.6', 'jupyterlab-pygments==0.3.0', 'opencv-python-headless==4.8.1.78', 'azure-mgmt-eventhub==10.1.0', 'watchfiles==0.20.0', 'iniconfig==2.0.0', 'azure-keyvault-secrets==4.7.0', 'django-extensions==3.2.3', 'torch==2.2.0', 'wasabi==1.1.1', 'mpmath==1.2.1', 'apache-airflow-providers-common-sql==1.11.0', 'gast==0.5.3', 'greenlet==3.0.2', 'bandit==1.7.6', 'pickleshare==0.7.3', 'webencodings==0.5', 'opentelemetry-exporter-otlp-proto-common==1.21.0', 'qtconsole==5.4.4', 'passlib==1.7.3', 'azure-mgmt-cosmosdb==9.4.0', 'jpype1==1.5.0', 'websockets==11.0.2', 'delta-spark==2.4.0', 'streamlit==1.32.2', 'murmurhash==1.0.9', 'azure-mgmt-netapp==11.0.0', 'awscli==1.32.64', 'nvidia-cudnn-cu11==8.9.6.50', 'catalogue==2.0.9', 'jaraco-classes==3.2.3', 'pydash==7.0.7', 'rfc3986==1.5.0', 'retrying==1.3.2', 'azure-mgmt-keyvault==10.3.0', 'azure-mgmt-iothubprovisioningservices==1.0.0', 'py4j==0.10.9.5', 'safetensors==0.4.0', 'astunparse==1.6.3', 'overrides==7.7.0', 'azure-storage-common==2.0.0', 'httplib2==0.21.0', 'distro==1.7.0', 'imagesize==1.3.0', 'anyio==4.2.0', 'google-cloud-videointelligence==2.13.2', 'ipython==8.18.0', 'google-cloud-videointelligence==2.13.1', 'httpx==0.25.2', 's3fs==2024.3.0', 'google-cloud-bigtable==2.22.0', 'argcomplete==3.2.3', 'datadog-api-client==2.22.0', 'sphinxcontrib-qthelp==1.0.7', 'openai==1.14.0', 'click-man==0.4.0', 'pyhcl==0.4.3', 'inflection==0.5.0', 'twisted==23.8.0', 'plotly==5.20.0', 'faker==24.2.0', 'universal-pathlib==0.2.2', 'types-redis==4.6.0.20240311', 'unittest-xml-reporting==3.0.4', 'azure-identity==1.14.0', 'jsonpath-ng==1.6.0', 'hyperlink==20.0.0', 'pytest-runner==5.3.2', 'typeguard==4.1.5', 'cog==0.9.5', 'py==1.9.0', 'timm==0.9.11', 'types-python-dateutil==2.9.0.20240316', 'azure-mgmt-synapse==1.0.0', 'markupsafe==2.1.3', 'magicattr==0.1.5', 'azure-mgmt-iothub==3.0.0', 'simplejson==3.19.2', 'spacy==3.7.1', 'azure-mgmt-servicefabric==1.0.0', 'astroid==3.1.0', 'sh==2.0.5', 'bitarray==2.9.0', 'azure-mgmt-imagebuilder==1.2.0', 'cloudpathlib==0.16.0', 'geographiclib==1.52', 'psycopg2-binary==2.9.7', 'pyrsistent==0.19.2', 'db-dtypes==1.1.0', 'pyflakes==3.2.0', 'cachetools==5.3.1', 'pyproj==3.5.0', 'google-cloud-datastore==2.17.0', 'shellingham==1.5.3', 'zope-event==5.0', 'pandas==2.2.1', 'bitarray==2.9.2', 'tensorflow-text==2.16.1', 'azure-mgmt-media==10.1.0', 'cramjam==2.8.1', 'langchain==0.1.12', 'jupyterlab==4.1.4', 'pyproject-api==1.6.0', 'xlwt==1.2.0', 'ansible-core==2.15.8', 'locket==0.2.0', 'ipython==8.17.2', 'bytecode==0.15.0', 'azure-keyvault-administration==4.4.0', 'cmake==3.28.3', 'djangorestframework==3.15.0', 'pyotp==2.8.0', 'inflect==6.2.0', 'flask==3.0.1', 'robotframework-seleniumlibrary==6.1.2', 'azure-mgmt-resource==23.0.1', 'validators==0.21.1', 'hyperframe==6.0.1', 'pytest-metadata==3.0.0', 'tifffile==2024.2.12', 'exceptiongroup==1.1.2', 'azure-mgmt-cosmosdb==9.2.0', 'click-didyoumean==0.1.0', 'typed-ast==1.5.4', 'gradio==4.21.0', 'thrift==0.16.0', 'redshift-connector==2.0.917', 'azure-cli==2.56.0', 'apache-airflow-providers-slack==8.6.0', 'llvmlite==0.41.0', 'selenium==4.18.0', 'apispec==6.4.0', 'aiosignal==1.2.0', 'azure-keyvault==4.1.0', 'ruamel-yaml-clib==0.2.7', 'aniso8601==8.1.1', 'dill==0.3.7', 'rfc3986==1.4.0', 'nvidia-cuda-nvrtc-cu12==12.3.107', 'apache-airflow-providers-ssh==3.10.1', 'pyrsistent==0.19.3', 'flask-babel==3.0.1', 'flask-jwt-extended==4.5.2', 'pytest-asyncio==0.23.3', 'promise==2.2', 'sentence-transformers==2.5.0', 'apache-airflow-providers-http==4.9.1', 'mypy==1.7.1', 'virtualenv==20.24.7', 'protobuf3-to-dict==0.1.4', 'funcsigs==1.0.1', 'xarray==2024.1.1', 'pika==1.3.1', 'pytest-asyncio==0.23.4', 'antlr4-python3-runtime==4.12.0', 'convertdate==2.4.0', 'parsedatetime==2.4', 'ftfy==6.1.1', 'commonmark==0.9.1', 'yarl==1.9.3', 'amqp==5.1.0', 'alembic==1.13.0', 'numba==0.58.1', 'uri-template==1.2.0', 'ipywidgets==8.1.0', 'qtconsole==5.5.1', 'google-re2==0.2.20220601', 'nvidia-cuda-nvrtc-cu12==12.3.103', 'tensorflow-probability==0.24.0', 'mock==5.1.0', 'google-auth==2.28.2', 'aiosignal==1.1.2', 'wheel==0.43.0', 'pgpy==0.6.0', 'beautifulsoup4==4.12.3', 'javaproperties==0.8.1', 'tensorflow-estimator==2.13.0', 'opencv-python-headless==4.9.0.80', 'databricks-api==0.9.0', 'djangorestframework==3.14.0', 'cfn-lint==0.86.0', 'multiprocess==0.70.16', 'azure-mgmt-batchai==1.0.1', 'bracex==2.3', 'python-daemon==2.3.0', 'decorator==5.1.0', 'stack-data==0.6.3', 'webdriver-manager==4.0.0', 'google-api-python-client==2.122.0', 'nbformat==5.10.1', 'hypothesis==6.99.4', 'tensorflow-serving-api==2.14.1', 'cymem==2.0.7', 'pyasn1==0.4.8', 'freezegun==1.4.0', 'pyasn1-modules==0.2.7', 'gitpython==3.1.42', 'pytimeparse==1.1.7', 'referencing==0.33.0', 'python-multipart==0.0.9', 'asyncio==3.4.1', 'pybind11==2.11.1', 'promise==2.2.1', 'apache-airflow-providers-common-sql==1.11.1', 'azure-mgmt-eventgrid==10.2.0', 'kr8s==0.13.6', 'imagesize==1.4.0', 'azure-synapse-artifacts==0.16.0', 'blinker==1.7.0', 'pytzdata==2019.2', 'pg8000==1.30.5', 'prometheus-flask-exporter==0.22.4', 'azure-mgmt-loganalytics==11.0.0', 'user-agent==0.1.10', 'filelock==3.13.0', 'pytimeparse==1.1.6', 'commonmark==0.9.0', 'tinycss2==1.2.1', 'huggingface-hub==0.21.3', 'google==3.0.0', 'chardet==5.2.0', 'astor==0.8.1', 'msgpack==1.0.8', 'agate==1.8.0', 'dnspython==2.6.0', 'azure-mgmt-applicationinsights==3.0.0', 'rfc3339-validator==0.1.3', 'retrying==1.3.3', 'tensorflow-datasets==4.9.1', 'fabric==3.2.2', 'evidently==0.4.17', 'blinker==1.6.2', 'future==1.0.0', 'django-extensions==3.2.1', 'torch==2.2.1', 'referencing==0.32.1', 'kombu==5.3.4', 'greenlet==3.0.3', 'nvidia-cufft-cu12==11.2.0.44', 'starkbank-ecdsa==2.2.0', 'apache-airflow-providers-databricks==6.1.0', 'pure-eval==0.2.2', 'asttokens==2.4.1', 'deepdiff==6.7.1', 'hyperframe==6.0.0', 'django-filter==24.1', 'json5==0.9.23', 'async-lru==2.0.4', 'zope-event==4.6', 'azure-mgmt-imagebuilder==1.1.0', 'azure-mgmt-iothubprovisioningservices==0.3.0', 'tensorstore==0.1.55', 'ndg-httpsclient==0.5.1', 'spacy-legacy==3.0.11', 'nvidia-cudnn-cu11==8.9.4.25', 'boto3-stubs==1.34.62', 'cattrs==23.2.3', 'dbt-postgres==1.7.9', 'ipython==8.18.1', 'netaddr==1.2.1', 'linkify-it-py==2.0.2', 'graphql-core==3.2.2', 'regex==2023.12.25', 'requests-mock==1.11.0', 'elasticsearch-dsl==8.11.0', 'tldextract==5.1.0', 'google-cloud-logging==3.8.0', 'cookiecutter==2.6.0', 'ftfy==6.1.3', 'hvac==1.2.1', 'voluptuous==0.14.2', 'logbook==1.5.3', 'paramiko==3.3.1', 'cssselect==1.0.3', 'evergreen-py==3.6.21', 'texttable==1.6.6', 'google-cloud-kms==2.21.2', 'datasets==2.17.1', 'uritemplate==4.1.1', 'nvidia-cufft-cu12==11.0.12.1', 'typeguard==4.1.3', 'sortedcontainers==2.2.2', 'cmake==3.28.1', 'convertdate==2.3.2', 'pyelftools==0.31', 'hiredis==2.2.3', 'qrcode==7.3.1', 'dateparser==1.1.7', 'azure-storage-common==2.1.0', 'google-cloud-appengine-logging==1.4.2', 'grpcio==1.62.1', 'django==4.2.11', 'azure-mgmt-search==9.0.0', 'wrapt==1.15.0', 'pyotp==2.7.0', 'ml-dtypes==0.3.2', 'flask-caching==2.0.2', 'pycparser==2.19', 'deprecated==1.2.12', 'simple-salesforce==1.12.3', 'pyyaml==6.0.1', 'opentelemetry-exporter-otlp-proto-common==1.23.0', 'asn1crypto==1.5.1', 'pbr==5.11.1', 'ply==3.11', 'keras-applications==1.0.6', 'gitdb==4.0.9', 'apache-airflow==2.8.1', 'apache-airflow-providers-databricks==6.0.0', 'jinja2==3.1.1', 'kafka-python==2.0.0', 'ua-parser==0.16.0', 'jira==3.5.2', 'azure-common==1.1.28', 'python-editor==1.0.4', 'sphinxcontrib-htmlhelp==2.0.3', 'jupyter-core==5.7.0', 'azure-mgmt-reservations==2.1.0', 'geographiclib==2.0', 'configparser==6.0.1', 'dacite==1.7.0', 'enum34==1.1.8', 'azure-storage-file-share==12.15.0', 'hdfs==2.7.2', 'importlib-metadata==7.0.2', 'google-cloud-datastore==2.18.0', 'ipywidgets==8.1.2', 'azure-datalake-store==0.0.52', 'jupyter-server==2.12.5', 'ipaddress==1.0.22', 'lightgbm==4.3.0', 'websocket-client==1.6.3', 'ansible==8.6.1', 'orbax-checkpoint==0.5.6', 'azure-graphrbac==0.61.0', 'editables==0.4', 'markdown==3.6', 'jsonschema-specifications==2023.12.1', 'fasteners==0.17.3', 'knack==0.10.1', 'cymem==2.0.6', 'keyring==24.2.0', 'azure-mgmt-security==5.0.0', 'accelerate==0.28.0', 'nvidia-curand-cu12==10.3.5.119', 'ijson==3.2.3', 'click==8.1.5', 'scandir==1.8', 'pyathena==3.3.0', 'azure-mgmt-applicationinsights==4.0.0', 'flask-session==0.6.0', 'passlib==1.7.2', 'pydantic-core==2.16.2', 'pyodbc==5.1.0', 'aws-xray-sdk==2.12.0', 'xlrd==2.0.0', 'python-slugify==8.0.4', 'types-requests==2.31.0.20240218', 'tblib==2.0.0', 'setuptools==69.2.0', 'delta-spark==3.0.0', 'jupyterlab-widgets==3.0.9', 'azure-mgmt-recoveryservicesbackup==9.0.0', 'netaddr==0.10.1', 'gradio-client==0.10.1', 'py==1.11.0', 'aws-xray-sdk==2.13.0', 'rsa==4.8', 'databricks-sql-connector==3.1.0', 'pylint==3.1.0', 'mypy-boto3-rds==1.34.58', 'libcst==1.2.0', 'types-s3transfer==0.8.2', 'azure-core==1.30.1', 'aws-psycopg2==1.2.0', 'entrypoints==0.4', 'contextlib2==0.5.5', 'gsutil==5.25', 'python-gitlab==4.4.0', 'termcolor==2.4.0', 'azure-keyvault-administration==4.2.0', 'pooch==1.8.0', 'opencv-python==4.8.0.76', 'wsproto==1.1.0', 'h2==3.1.1', 'azure-keyvault-keys==4.9.0', 'grpcio-health-checking==1.62.1', 'parse-type==0.6.2', 'overrides==7.6.0', 'azure-synapse-spark==0.5.0', 'azure-eventgrid==4.18.0', 'sphinx==7.2.5', 'ujson==5.7.0', 'jsonschema-specifications==2023.11.2', 'imageio==2.33.1', 'azure-mgmt-datalake-store==0.5.0', 'evidently==0.4.15', 'azure-data-tables==12.5.0', 'pytz==2024.1', 'nvidia-cuda-nvrtc-cu12==12.4.99', 'ua-parser==0.18.0', 'imageio==2.33.0', 'datadog-api-client==2.21.0', 'constructs==10.3.0', 'snowballstemmer==2.2.0', 'jupyterlab-widgets==3.0.10', 'spacy-loggers==1.0.4', 'agate==1.9.0', 'apache-airflow-providers-cncf-kubernetes==8.0.1', 'dbt-extractor==0.5.0', 'jupyter-lsp==2.2.3', 'ml-dtypes==0.3.1', 'sphinx-rtd-theme==1.2.2', 'argcomplete==3.2.2', 'azure-data-tables==12.4.3', 'pathlib==1.0', 'scikit-learn==1.3.2', 'tensorflow-metadata==1.13.1', 'requests-aws4auth==1.2.1', 'tabulate==0.9.0', 'exceptiongroup==1.2.0', 'pysftp==0.2.8', 'jsonpatch==1.31', 'docopt==0.6.1', 'aenum==3.1.15', 'yarl==1.9.2', 'office365-rest-python-client==2.5.5', 'psycopg==3.1.18', 'azure-kusto-data==4.3.0', 'retry==0.8.1', 'lark==1.1.8', 'imagesize==1.4.1', 'wandb==0.16.3', 'twilio==8.12.0', 'ptyprocess==0.7.0', 'pathspec==0.12.0', 'azure-mgmt-signalr==1.1.0', 'threadpoolctl==3.3.0', 'django-cors-headers==4.3.0', 'schema==0.7.5', 'nltk==3.8.1', 'nvidia-cudnn-cu12==8.9.6.50', 'ruff==0.3.1', 'spacy-loggers==1.0.3', 'jedi==0.18.2', 'coloredlogs==14.3', 'atomicwrites==1.4.0', 'types-pytz==2023.3.1.1', 'uc-micro-py==1.0.1', 'azure-mgmt-appconfiguration==3.0.0', 'argparse==1.4.0', 'autopep8==2.0.2', 'ipywidgets==8.1.1', 'click-repl==0.3.0', 'azure-mgmt-recoveryservices==2.4.0', 'google-crc32c==1.5.0', 'sphinxcontrib-qthelp==1.0.5', 'opentelemetry-sdk==1.22.0', 'snowballstemmer==2.1.0', 'python-magic==0.4.27', 'apache-airflow-providers-ftp==3.6.1', 'iso8601==1.1.0', 'hypothesis==6.99.6', 'opt-einsum==3.3.0', 'azure-servicebus==7.11.3', 'smmap==4.0.0', 'readme-renderer==43.0', 'absl-py==2.1.0', 'azure-mgmt-kusto==3.2.0', 'looker-sdk==24.2.0', 'flit-core==3.9.0', 'pyspark==3.4.2', 'grpcio-health-checking==1.60.1', 'watchdog==3.0.0', 'hypothesis==6.99.5', 'jsonschema==4.20.0', 'pg8000==1.30.4', 'azure-servicebus==7.11.4', 'querystring-parser==1.2.3', 'mdit-py-plugins==0.3.5', 'asgiref==3.6.0', 'azure-storage-blob==12.19.1', 'jupyter-events==0.8.0', 'applicationinsights==0.11.10', 'joblib==1.3.2', 'watchtower==3.1.0', 'mlflow==2.11.1', 'aiodns==3.0.0', 'nbformat==5.10.2', 'atomicwrites==1.3.0', 'inflection==0.5.1', 'langsmith==0.1.25', 'fqdn==1.4.0', 'flit-core==3.7.1', 'numexpr==2.8.6', 'debugpy==1.8.0', 'mako==1.2.4', 'colorlog==6.7.0', 'fastapi==0.109.1', 'jsonpath-ng==1.6.1', 'graphviz==0.20', 'djangorestframework==3.13.1', 'shap==0.45.0', 'google-cloud-pubsub==2.20.1', 'pure-eval==0.2.0', 'pytest-rerunfailures==14.0', 'azure-mgmt-hdinsight==8.0.0', 'tiktoken==0.6.0', 'py4j==0.10.9.6', 'azure-mgmt-containerservice==29.0.0', 'markupsafe==2.1.5', 'azure-eventhub==5.11.4', 'opentelemetry-exporter-otlp-proto-http==1.23.0', 'schema==0.7.4', 'timm==0.9.16', 'mashumaro==3.12', 'murmurhash==1.0.10', 'coloredlogs==15.0', 'jupyter-client==8.6.0', 'kiwisolver==1.4.5', 'flask-cors==3.0.9', 'packaging==23.1', 'scikit-learn==1.3.1', 'cx-oracle==8.3.0', 'oauthlib==3.2.0', 'webcolors==1.11.1', 'aioitertools==0.11.0', 'azure-mgmt-devtestlabs==3.0.0', 'sh==2.0.6', 'commonmark==0.8.1', 'sphinxcontrib-devhelp==1.0.5', 'srsly==2.4.8', 'opensearch-py==2.4.1', 'referencing==0.34.0', 'azure-storage-file-datalake==12.13.2', 'nh3==0.2.14', 'azure-mgmt-devtestlabs==9.0.0', 'opentelemetry-proto==1.21.0', 'jupyter-server==2.13.0', 'h2==3.2.0', 'pywavelets==1.3.0', 'azure-mgmt-containerservice==29.1.0', 'lz4==4.3.2', 'libcst==1.0.1', 'azure-synapse-accesscontrol==0.5.0', 'apache-airflow-providers-ftp==3.7.0', 'plotly==5.19.0', 'dbt-extractor==0.4.1', 'multiprocess==0.70.14', 'azure-mgmt-monitor==6.0.2', 'termcolor==2.3.0', 'azure-synapse-spark==0.7.0', 'msal==1.26.0', 'flask==3.0.0', 'babel==2.13.1', 'kombu==5.3.5', 'nvidia-cuda-cupti-cu12==12.3.101', 'dvc-render==0.7.0', 'holidays==0.42', 'configobj==5.0.5', 'robotframework-seleniumlibrary==6.1.3', 'sphinxcontrib-applehelp==1.0.6', 'pyflakes==3.0.1', 'rdflib==7.0.0', 'msal==1.25.0', 'dacite==1.8.1', 'boto==2.48.0', 'azure-cli-telemetry==1.1.0', 'xyzservices==2023.7.0', 'accelerate==0.27.2', 'python-dateutil==2.8.2', 'python-docx==1.0.1', 'pywavelets==1.5.0', 'ipaddress==1.0.21', 'pytest-runner==6.0.1', 'pyarrow==14.0.2', 'bokeh==3.3.4', 'msrestazure==0.6.4', 'appdirs==1.4.4', 'apache-airflow-providers-ssh==3.10.0', 'flask-wtf==1.2.0', 'hologram==0.0.14', 'asynctest==0.12.4', 'makefun==1.15.1', 'python-slugify==8.0.3', 'nvidia-cuda-cupti-cu12==12.3.52', 'azure-mgmt-netapp==10.0.0', 'importlib-metadata==7.0.0', 'flask-jwt-extended==4.6.0', 'billiard==4.2.0', 'jsonpickle==3.0.0', 'jsonpickle==3.0.1', 'flatbuffers==23.5.26', 'tensorboard-data-server==0.7.1', 'zope-interface==6.2', 'requests-file==1.5.1', 'unittest-xml-reporting==3.2.0', 'marshmallow==3.21.1', 'idna==3.5', 'mpmath==1.1.0', 'future==0.18.2', 'notebook-shim==0.2.3', 'google==2.0.3', 'aiohttp==3.9.2', 'limits==3.10.0', 'mkdocs-material==9.5.11', 'sympy==1.12', 'google-auth-oauthlib==1.0.0', 'azure-mgmt-recoveryservices==2.5.0', 'pathos==0.3.2', 'apache-airflow-providers-databricks==6.2.0', 'telethon==1.34.0', 'preshed==3.0.7', 'cinemagoer==2022.12.4', 'types-awscrt==0.20.4', 'azure-storage-file-share==12.14.2', 'types-pyyaml==6.0.12.11', 'mashumaro==3.11', 'azure-core==1.30.0', 'mmh3==4.1.0', 'validators==0.22.0', 'spark-nlp==5.3.0', 'db-contrib-tool==0.6.13', 'gradio==4.20.0', 'types-pytz==2024.1.0.20240203', 'dataclasses-json==0.6.2', 'office365-rest-python-client==2.5.4', 'ultralytics==8.1.27', 'simple-salesforce==1.12.5', 'slicer==0.0.8', 'apache-airflow-providers-slack==8.6.1', 'orbax-checkpoint==0.5.4', 'pycryptodome==3.20.0', 'python-magic==0.4.26', 'mypy==1.9.0', 'google-cloud-monitoring==2.19.3', 'charset-normalizer==3.3.0', 'lightgbm==4.1.0', 'execnet==2.0.2', 'azure-graphrbac==0.61.1', 'sortedcontainers==2.4.0', 'linkify-it-py==2.0.1', 'retrying==1.3.4', 'rsa==4.9', 'starlette==0.37.2', 'gevent==24.2.1', 'google-cloud-language==2.13.3', 'jupyter-server==2.12.4', 'user-agents==2.0', 'poetry-plugin-export==1.7.0', 'grpcio-health-checking==1.62.0', 'factory-boy==3.3.0', 'pycryptodomex==3.19.1', 'markdown==3.5.2', 'addict==2.3.0', 'ptyprocess==0.5.2', 'azure-cli==2.57.0', 'retry==0.7.0', 'azure-mgmt-datalake-store==0.4.0', 'fabric==3.2.0', 'kafka-python==2.0.1', 'pooch==1.7.0', 'jsonpointer==2.4', 'nbclassic==0.5.5', 'identify==2.5.34', 'openapi-spec-validator==0.7.1', 'azure-mgmt-sql==3.0.0', 'google-cloud-aiplatform==1.42.1', 'msrest==0.6.19', 'unicodecsv==0.14.0', 'spacy-loggers==1.0.5', 'docstring-parser==0.16', 'pyee==11.0.1', 'deepdiff==6.6.1', 'seaborn==0.13.0', 'distrax==0.1.3', 'croniter==2.0.2', 'widgetsnbextension==4.0.10', 'validators==0.21.2', 'nodeenv==1.8.0', 'azure-mgmt-batch==17.2.0', 'trove-classifiers==2024.3.3', 'slack-sdk==3.27.1', 'nvidia-cuda-runtime-cu12==12.4.99', 'google-cloud-language==2.13.2', 'tzlocal==5.2', 'azure-storage-blob==12.19.0', 'azure-mgmt-recoveryservicesbackup==8.0.0', 'nvidia-cusparse-cu12==12.3.0.142', 'aioitertools==0.10.0', 'rsa==4.7.2', 'tiktoken==0.5.1', 'azure-datalake-store==0.0.51', 'types-s3transfer==0.9.0', 'pytest==8.0.1', 'pillow==10.0.1', 'isort==5.13.0', 'spacy-legacy==3.0.12', 'amqp==5.1.1', 'wasabi==0.10.1', 'azure-mgmt-datamigration==4.1.0', 'oldest-supported-numpy==2023.12.21', 'ordered-set==4.0.1', 'hpack==3.0.0', 'backoff==2.2.1', 'datadog-api-client==2.23.0', 'traitlets==5.14.2', 'zeep==4.2.0', 'requests-ntlm==1.2.0', 'elasticsearch==8.11.1', 'pox==0.3.2', 'thinc==8.2.2', 'holidays==0.44', 'opentelemetry-sdk==1.23.0', 'typing==3.7.4.3', 'azure-mgmt-iothub==2.3.0', 'toolz==0.11.2', 'sphinxcontrib-devhelp==1.0.4', 'azure-mgmt-recoveryservices==2.3.0', 'gunicorn==21.0.1', 'google-cloud-bigquery-storage==2.24.0', 'python-slugify==8.0.2', 'typer==0.9.0', 'pyspark==3.5.1', 'ansible==8.6.0', 'kfp-pipeline-spec==0.3.0', 'sympy==1.11', 'kubernetes==27.2.0', 'bcrypt==4.1.1', 'keyring==24.3.0', 'itsdangerous==2.1.2', 'beautifulsoup4==4.12.1', 'pandocfilters==1.4.3', 'pytest-mock==3.11.0', 'dulwich==0.21.6', 'google-cloud-audit-log==0.2.5', 'gremlinpython==3.7.0', 'flatbuffers==24.3.7', 'azure-mgmt-applicationinsights==3.1.0', 'jaydebeapi==1.2.3', 'pytz==2023.4', 'pyee==11.0.0', 'dataclasses==0.4', 'soupsieve==2.4.1', 'invoke==2.1.3', 'azure-mgmt-monitor==6.0.1', 'zipp==3.18.1', 'python-http-client==3.3.7', 'google-ads==22.1.0', 'azure-mgmt-containerinstance==10.0.0', 'constructs==10.2.70', 'pydantic==2.6.4', 'slackclient==2.9.3', 'pg8000==1.30.3', 'applicationinsights==0.11.9', 'azure-mgmt-maps==1.0.0', 'tldextract==5.1.1', 'cffi==1.15.0', 'seaborn==0.13.1', 'shap==0.44.1', 'google-resumable-media==2.5.0', 'aws-lambda-powertools==2.35.0', 'apache-airflow-providers-ftp==3.6.0', 'pypdf2==3.0.0', 'setproctitle==1.3.3', 'ansible==8.7.0', 'grpcio-tools==1.62.1', 'cached-property==1.5.2', 'beautifulsoup4==4.12.2', 'pyperclip==1.8.1', 'parse==1.20.0', 'outcome==1.1.0', 'polars==0.20.13', 'user-agent==0.1.9', 'uvloop==0.17.0', 'gast==0.5.2', 'pysftp==0.2.7', 'pymssql==2.2.9', 'babel==2.13.0', 'pytest-forked==1.6.0', 'azure-mgmt-loganalytics==12.0.0', 'alabaster==0.7.14', 'joblib==1.3.0', 'python-editor==1.0.3', 'docutils==0.19', 'pydot==1.4.1', 'greenlet==3.0.1', 'text-unidecode==1.2', 'setproctitle==1.3.2', 'wcwidth==0.2.11', 'smmap==5.0.1', 'frozenlist==1.4.1', 'lightning-utilities==0.10.0', 'jupyterlab==4.1.3', 'feedparser==6.0.9', 'flask-session==0.5.0', 'grpcio-tools==1.60.1', 'unicodecsv==0.14.1', 'maxminddb==2.5.0', 'parso==0.8.1', 'importlib-resources==6.3.0', 'nbformat==5.10.3', 'watchfiles==0.19.0', 'uc-micro-py==1.0.3', 'yapf==0.40.1', 'pyopenssl==24.1.0', 'google-cloud-bigquery-storage==2.22.0', 'h11==0.12.0', 'fsspec==2024.3.0', 'secretstorage==3.3.1', 'pytest-xdist==3.5.0', 'django-cors-headers==4.2.0', 'fastavro==1.9.4', 'distrax==0.1.5', 'xyzservices==2023.10.1', 'safetensors==0.4.2', 'six==1.15.0', 'dacite==1.8.0', 'aniso8601==9.0.1', 'sentence-transformers==2.4.0', 'future==0.18.3', 'scp==0.14.3', 'deprecated==1.2.13', 'python-jose==3.2.0', 'pydeequ==1.2.0', 'kfp==2.6.0', 'pytest-timeout==2.3.1', 'marshmallow-enum==1.5.1', 'onnxruntime==1.17.0', 'webdriver-manager==3.9.1', 'xmltodict==0.12.0', 'nvidia-nccl-cu12==2.18.3', 'fsspec==2024.2.0', 'sniffio==1.2.0', 'levenshtein==0.24.0', 'google-api-python-client==2.120.0', 'simple-salesforce==1.12.4', 'typing-extensions==4.10.0', 'prometheus-flask-exporter==0.22.3', 'grpcio==1.62.0', 'pytest-timeout==2.1.0', 'azure-mgmt-apimanagement==4.0.0', 'phonenumbers==8.13.32', 'jaraco-classes==3.3.1', 'apache-airflow-providers-amazon==8.18.0', 'configupdater==3.1', 'azure-mgmt-loganalytics==10.0.0', 'cython==3.0.9', 'imdbpy==2022.7.9', 'lockfile==0.10.2', 'confluent-kafka==2.1.1', 'timm==0.9.12', 'webcolors==1.12', 'stringcase==1.0.6', 'distributed==2024.3.0', 'dm-tree==0.1.8', 'unidecode==1.3.8', 'google-cloud-container==2.42.0', 'levenshtein==0.23.0', 'trio==0.23.2', 'cachetools==5.3.3', 'cog==0.9.4', 'semantic-version==2.8.5', 'factory-boy==3.2.0', 'emoji==2.9.0', 'geopandas==0.14.3', 'dulwich==0.21.5', 'hyperlink==21.0.0', 'ddtrace==2.7.1', 'wheel==0.42.0', 'astunparse==1.6.2', 'wrapt==1.16.0', 'tenacity==8.2.2', 'frozenlist==1.4.0', 'databricks-cli==0.18.0', 'ansible-core==2.15.7', 'pooch==1.8.1', 'azure-mgmt-reservations==2.2.0', 'pyopenssl==23.3.0', 'authlib==1.2.1', 'isort==5.13.2', 'tensorboard-data-server==0.7.0', 'fire==0.5.0', 'gcsfs==2023.12.0', 'shortuuid==1.0.12', 'azure-mgmt-maps==2.1.0', 'trove-classifiers==2024.2.22', 'filelock==3.13.1', 'azure-mgmt-botservice==1.0.0', 'xarray==2024.1.0', 'termcolor==2.2.0', 'opencensus==0.11.4', 'azure-mgmt-msi==7.0.0', 'flask-cors==3.0.10', 'alabaster==0.7.15', 'pyspnego==0.9.1', 'pika==1.3.0', 'requests-aws4auth==1.2.2', 'pytzdata==2020.1', 'jsonpatch==1.33', 'typing-extensions==4.9.0', 'autopep8==2.0.4', 'platformdirs==4.2.0', 'azure-mgmt-synapse==2.0.0', 'oauth2client==4.1.2', 'psycopg2-binary==2.9.9', 'executing==2.0.1', 'ruamel-yaml==0.18.6', 'azure-cosmos==4.5.1', 'holidays==0.43', 'text-unidecode==1.3', 'graphviz==0.19.2', 'gsutil==5.26', 'poetry-plugin-export==1.6.0', 'pathlib2==2.3.6', 'rich==13.7.0', 'cookiecutter==2.4.0', 'google-cloud-monitoring==2.19.1', 'azure-mgmt-netapp==10.1.0', 'typing-inspect==0.7.1', 'google-cloud-secret-manager==2.18.1', 'poetry-core==1.8.0', 'sentencepiece==0.2.0', 'setuptools==69.1.1', 'azure-storage-queue==12.9.0', 'pytest-html==4.1.0', 'imdbpy==2020.9.25', 'pyaml==23.9.7', 'tableauserverclient==0.28', 'python-json-logger==2.0.5', 'poetry-core==1.9.0', 'apache-airflow-providers-ssh==3.9.0', 'tensorflow-hub==0.16.1', 'azure-mgmt-botservice==2.0.0', 'httpcore==1.0.2', 'langchain==0.1.11', 'jpype1==1.4.1', 'keras-preprocessing==1.1.0', 'redis==5.0.1', 'jupyter-events==0.9.1', 'python-box==7.1.0', 'requests-mock==1.10.0', 'azure-eventhub==5.11.5', 'avro==1.11.2', 'python-docx==1.0.0', 'incremental==22.10.0', 'nest-asyncio==1.6.0', 'typing==3.7.4.1', 'build==1.0.0', 'watchfiles==0.21.0', 'confluent-kafka==2.2.0', 'azure-mgmt-datafactory==5.0.0', 'types-pyyaml==6.0.12.12', 'mlflow==2.11.0', 'adal==1.2.6', 'six==1.16.0', 'ninja==1.11.1', 'hyperlink==20.0.1', 'xlrd==2.0.1', 'blinker==1.6.3', 'azure-mgmt-containerinstance==10.1.0', 'pymysql==1.0.2', 'azure-mgmt-datalake-store==0.3.0', 'yamllint==1.35.0', 'vine==5.1.0', 'black==24.1.1', 'requests-toolbelt==0.10.1', 'jupyterlab-pygments==0.2.1', 'evaluate==0.4.0', 'selenium==4.17.2', 'sphinxcontrib-applehelp==1.0.7', 's3transfer==0.9.0', 'fastjsonschema==2.18.1', 'pydub==0.25.1', 'prettytable==3.9.0', 'pygithub==2.2.0', 'configobj==5.0.6', 'cached-property==1.4.3', 'azure-mgmt-appconfiguration==2.2.0', 'python-utils==3.8.1', 'marshmallow-enum==1.4', 'pytest-xdist==3.4.0', 'awswrangler==3.7.0', 'google-cloud-tasks==2.16.1', 'databricks-cli==0.17.8', 'gspread==6.0.1', 'celery==5.3.6', 'aiodns==3.1.1', 'nvidia-cuda-cupti-cu12==12.4.99', 'semver==3.0.2', 'chex==0.1.85', 'django-filter==23.4', 'cx-oracle==8.2.0', 'msrest==0.6.21', 'django-extensions==3.2.0', 'python-utils==3.8.2', 'argcomplete==3.2.1', 'azure-servicebus==7.12.0', 'boto3==1.34.62', 'google-cloud-dlp==3.16.0', 'pypdf2==3.0.1', 'notebook==7.1.2', 'db-contrib-tool==0.6.14', 'httptools==0.6.0', 'aiobotocore==2.11.2', 'typing-inspect==0.8.0', 'azure-eventgrid==4.16.0', 'qrcode==7.4.2', 'yapf==0.40.2', 'jinja2==3.1.2', 'azure-mgmt-policyinsights==0.6.0', 'azure-synapse-spark==0.6.0', 'types-python-dateutil==2.9.0.20240315', 'azure-datalake-store==0.0.53', 'wrapt==1.14.1', 'apache-airflow==2.8.2', 'azure-mgmt-marketplaceordering==0.2.1', 'service-identity==21.1.0', 'flask-caching==2.1.0', 'prison==0.1.3', 'certifi==2023.11.17', 'rfc3986==2.0.0', 'python-levenshtein==0.24.0', 'pydantic-core==2.16.3', 'jsondiff==2.0.0', 'sendgrid==6.10.0', 'google-auth-oauthlib==1.2.0', 'tensorflow-serving-api==2.14.0', 'onnx==1.14.1', 'google-crc32c==1.3.0', 'tensorboard==2.16.2', 'spacy==3.7.4', 'boto==2.49.0', 'gradio-client==0.12.0', 'docstring-parser==0.15', 'outcome==1.3.0', 'azure-mgmt-search==8.0.0', 'hatchling==1.22.2', 'evaluate==0.3.0', 'azure-mgmt-eventgrid==10.0.0', 'more-itertools==10.0.0', 'ujson==5.8.0', 'tornado==6.3.2', 'azure-mgmt-apimanagement==3.0.0', 'pydot==1.4.2', 'rapidfuzz==3.6.0', 'phonenumbers==8.13.30', 'azure-keyvault-secrets==4.8.0', 'contourpy==1.1.1', 'google-auth-httplib2==0.1.0', 'aiodns==3.1.0', 'pox==0.3.4', 'httpcore==1.0.4', 'cffi==1.15.1', 'snowflake-connector-python==3.7.0', 'kfp==2.7.0', 'wandb==0.16.2', 'urllib3==2.2.0', 'mypy-boto3-redshift-data==1.33.0', 'docopt==0.6.2', 'nvidia-nvtx-cu12==12.3.101', 'snowflake-sqlalchemy==1.5.1', 'pyparsing==3.1.1', 'redis==5.0.2', 'appdirs==1.4.3', 'pathy==0.10.3', 'cdk-nag==2.28.66', 'sqlalchemy-utils==0.41.0', 'tzdata==2023.4', 'azure-graphrbac==0.60.0', 'datadog==0.48.0', 'azure-batch==14.0.0', 'user-agents==2.2.0', 'seaborn==0.13.2', 'transformers==4.38.0', 'google-cloud-storage==2.14.0', 'mock==5.0.2', 'cramjam==2.8.2', 'apache-airflow-providers-http==4.10.0', 'addict==2.2.1', 'cerberus==1.3.5', 'click-man==0.4.1', 'google-cloud-vision==3.7.1', 'google-cloud-core==2.3.3', 'azure-mgmt-web==7.1.0', 'toml==0.10.0', 'lazy-object-proxy==1.10.0', 'paramiko==3.3.0', 'json5==0.9.22', 'cron-descriptor==1.4.3', 'watchdog==2.3.1', 'azure-common==1.1.26', 'azure-mgmt-core==1.3.1', 'stevedore==5.2.0', 'websockets==11.0.3', 'pyasn1==0.5.0', 'argon2-cffi==21.2.0', 'tifffile==2024.1.30', 'pymongo==4.6.2', 'natsort==8.3.0', 'azure-mgmt-media==10.2.0', 'snowflake-sqlalchemy==1.5.0', 'ec2-metadata==2.12.0', 'python-gitlab==4.3.0', 'pyaml==23.12.0', 'pymeeus==0.5.11', 'ddtrace==2.7.2', 'jupyter-server-terminals==0.5.2', 'google-cloud-datacatalog==3.18.1', 'scipy==1.11.3', 'adal==1.2.5', 's3fs==2023.12.2', 'azure-mgmt-marketplaceordering==1.1.0', 'threadpoolctl==3.1.0', 'azure-common==1.1.27', 'pluggy==1.4.0', 'opentelemetry-exporter-otlp==1.21.0', 'cryptography==42.0.5', 'watchtower==3.0.0', 'redshift-connector==2.1.0', 'transformers==4.38.1', 'asyncio==3.4.3', 'lark==1.1.7', 'sagemaker==2.213.0', 'xlsxwriter==3.1.9', 'pyathena==3.2.1', 'jupyter-server-terminals==0.5.1', 'adal==1.2.7', 'azure-mgmt-servicefabric==2.1.0', 'ply==3.10', 'jaxlib==0.4.25', 'nvidia-nvtx-cu12==12.4.99', 'python-json-logger==2.0.7', 'opentelemetry-api==1.21.0', 'azure-mgmt-containerservice==28.0.0', 'azure-mgmt-servicebus==8.2.0', 'typed-ast==1.5.3', 'idna==3.4', 'sqlalchemy==2.0.26', 'celery==5.3.5', 'sagemaker==2.211.0', 'poetry==1.8.2', 'humanfriendly==10.0', 'structlog==23.2.0', 'httplib2==0.20.4', 'progressbar2==4.4.2', 'asgiref==3.7.1', 'azure-appconfiguration==1.4.0', 'click-plugins==1.1.1', 'google-cloud-audit-log==0.2.4', 'knack==0.11.0', 'time-machine==2.14.0', 'lxml==5.1.0', 'datadog==0.49.0', 'asynctest==0.13.0', 'googleapis-common-protos==1.61.0', 'jaxlib==0.4.23', 'send2trash==1.7.1', 'azure-storage-blob==12.18.3', 'emoji==2.10.1', 'azure-mgmt-appconfiguration==2.1.0', 'databricks-sdk==0.21.0', 'dbt-postgres==1.7.10', 'httptools==0.6.1', 'oauthlib==3.2.1', 'cleo==2.0.0', 'snowflake-connector-python==3.6.0', 'poetry-plugin-export==1.5.0', 'marshmallow-dataclass==8.6.0', 'fuzzywuzzy==0.16.0', 'arrow==1.2.2', 'azure-storage-queue==12.7.3', 'python-box==7.0.1', 'mysql-connector-python==8.2.0', 'boto3==1.34.64', 'six==1.14.0', 'absl-py==1.4.0', 'tiktoken==0.5.2', 'apache-airflow-providers-imap==3.4.0', 'pyasn1-modules==0.3.0', 'coverage==7.4.2', 'secretstorage==3.3.3', 'awscli==1.32.62', 'annotated-types==0.5.0', 'ruff==0.3.3', 'python-dateutil==2.8.1', 'py-cpuinfo==9.0.0', 'azure-mgmt-keyvault==10.2.3', 'smdebug-rulesconfig==1.0.0', 'argon2-cffi==21.3.0', 'docutils==0.20', 'jsonschema==4.21.0', 'reportlab==4.1.0', 'toml==0.10.1', 'sagemaker==2.212.0', 'cachecontrol==0.14.0', 'kiwisolver==1.4.4', 'google-cloud-bigquery==3.19.0', 'sphinxcontrib-applehelp==1.0.8', 'dpath==2.1.4', 'tabulate==0.8.9', 'google-cloud-secret-manager==2.18.3', 'google-cloud-firestore==2.15.0', 'dnspython==2.6.1', 'gspread==6.0.0', 'pydata-google-auth==1.8.1', 'send2trash==1.8.0', 'logbook==1.6.0', 'apache-airflow-providers-imap==3.5.0', 'python-dotenv==1.0.1', 'google-api-core==2.17.1', 'enum34==1.1.10', 'pyzmq==25.1.0', 'elasticsearch==8.12.1', 'azure-keyvault-certificates==4.8.0', 'azure-mgmt-rdbms==9.1.0', 'opentelemetry-api==1.23.0', 'dbt-core==1.7.8', 'datetime==5.3', 'asn1crypto==1.4.0', 'universal-pathlib==0.2.0', 'google-pasta==0.1.7', 'protobuf==5.26.0', 'msal==1.27.0', 'pathos==0.3.0', 'pika==1.3.2', 'pymongo==4.6.1', 'semantic-version==2.10.0', 'tensorflow-estimator==2.14.0', 'omegaconf==2.2.3', 'pyyaml==6.0', 'pluggy==1.2.0', 'types-redis==4.6.0.20240106', 'magicattr==0.1.6', 'netaddr==1.1.0', 'gensim==4.3.1', 'jax==0.4.24', 'pymeeus==0.5.10', 'azure-mgmt-rdbms==10.1.0', 'dbt-core==1.7.10', 'mako==1.3.2', 'ddsketch==2.0.2', 'elastic-transport==8.12.0', 'jupyterlab-server==2.25.2', 'firebase-admin==6.4.0', 'traitlets==5.14.1', 'chardet==5.1.0', 'apscheduler==3.10.4', 'google-cloud-build==3.23.3', 'flax==0.8.1', 'prometheus-client==0.20.0', 'plotly==5.18.0', 'azure-mgmt-batchai==1.0.0', 'pytest-rerunfailures==12.0', 'twisted==24.3.0', 'jeepney==0.7.0', 'tensorflow-probability==0.22.1', 'prettytable==3.8.0', 'charset-normalizer==3.3.2', 'psutil==5.9.7', 'openpyxl==3.1.1', 'types-urllib3==1.26.25.13', 'wsproto==1.2.0', 'zstandard==0.20.0', 'azure-keyvault-certificates==4.7.0', 'pytorch-lightning==2.1.4', 'azure-mgmt-iotcentral==4.1.0', 'dateparser==1.2.0', 'pyjwt==2.8.0', 'azure-storage-common==1.4.2', 'build==1.1.1', 'azure-mgmt-storage==21.0.0', 'cfgv==3.3.1', 'flatbuffers==24.3.6', 'msal-extensions==1.0.0', 'httplib2==0.22.0', 'colorama==0.4.6', 'libclang==16.0.6', 'korean-lunar-calendar==0.2.0', 'tensorboard-plugin-wit==1.8.0', 'kfp-server-api==2.0.4', 'paramiko==3.4.0', 'shortuuid==1.0.13', 'azure-mgmt-redis==14.3.0', 'checkov==3.2.37', 'nvidia-cusolver-cu12==11.5.4.101', 'mypy-extensions==0.4.3', 'pytest-runner==6.0.0', 'jeepney==0.8.0', 'cloudpathlib==0.17.0', 'xgboost==2.0.3', 'click-man==0.3.0', 'schema==0.7.2', 'zipp==3.17.0', 'redis==5.0.3', 'semver==3.0.1', 'msal-extensions==0.3.1', 'nose==1.3.7', 'botocore==1.34.62', 'toolz==0.12.0', 'build==1.0.3', 'aws-lambda-powertools==2.34.2', 'fire==0.6.0', 'pytzdata==2019.3', 'pybind11==2.10.4', 'pycryptodome==3.19.1', 'deprecation==2.0.6', 'autopep8==2.0.3', 'wtforms==3.1.0', 'mypy-boto3-appflow==1.34.0', 'pysocks==1.7.1', 'nvidia-nvjitlink-cu12==12.4.99', 'user-agent==0.1.8', 'html5lib==1.0.1', 'fastjsonschema==2.19.0', 'cachelib==0.11.0', 'pyyaml==5.4.1', 'jsondiff==1.3.0', 'google-cloud-core==2.4.0', 'rpds-py==0.18.0', 'opentelemetry-api==1.22.0', 'mashumaro==3.10', 'pyproj==3.6.0', 'azure-mgmt-consumption==9.0.0', 'dm-tree==0.1.6', 'async-timeout==4.0.3', 'dpath==2.1.5', 'cerberus==1.3.3', 'gql==3.4.1', 'soupsieve==2.4', 'jsonlines==3.0.0', 'convertdate==2.3.1', 'sqlalchemy-jsonfield==1.0.2', 'psycopg==3.1.16', 'unidecode==1.3.6', 'opentelemetry-exporter-otlp-proto-grpc==1.21.0', 'sqlalchemy-utils==0.40.0', 'defusedxml==0.6.0', 'onnx==1.15.0', 'virtualenv==20.25.0', 'pytest-cov==4.0.0', 'structlog==23.3.0', 'nvidia-nccl-cu12==2.19.3', 'pyproj==3.6.1', 'tensorboard==2.16.0', 'awscli==1.32.63', 'emoji==2.10.0', 'lazy-object-proxy==1.9.0', 'dvc-render==1.0.0', 'blis==0.9.0', 'python-levenshtein==0.25.0', 'hdfs==2.7.3', 'astroid==3.0.2', 'ansible-core==2.15.9', 'google-cloud-kms==2.21.1', 'python-dotenv==1.0.0', 'pep517==0.13.1', 'google-cloud-firestore==2.13.1', 'iniconfig==1.1.0', 'pandas==2.1.4', 'texttable==1.6.7', 'google-cloud-datacatalog==3.18.3', 'snowflake-sqlalchemy==1.4.7', 'typing-extensions==4.8.0', 'chex==0.1.84', 'trio-websocket==0.11.1', 'lark==1.1.9', 'invoke==2.1.2', 'pyasn1==0.5.1', 'defusedxml==0.7.0', 'proto-plus==1.22.3', 'sphinxcontrib-serializinghtml==1.1.10', 'apache-airflow-providers-slack==8.5.1', 'python-jose==3.3.0', 'humanize==4.8.0', 'makefun==1.15.0', 'colorama==0.4.5', 'packaging==24.0', 'types-requests==2.31.0.20240311', 'sniffio==1.3.1', 'urllib3==2.2.1', 'configargparse==1.5.5', 'lockfile==0.11.0', 'spark-nlp==5.3.1', 'jmespath==0.10.0', 'pyproject-api==1.6.1', 'marshmallow-sqlalchemy==0.29.0', 'unicodecsv==0.13.0', 'nvidia-nvtx-cu12==12.3.52', 'addict==2.4.0', 'tqdm==4.66.2', 'pyasn1-modules==0.2.8', 'pymssql==2.2.10', 'uamqp==1.6.8', 'time-machine==2.12.0', 'partd==1.4.0', 'pygithub==1.59.1', 'llvmlite==0.41.1', 'execnet==2.0.0', 'grpc-google-iam-v1==0.12.6', 'azure-mgmt-storage==20.1.0', 'pycryptodomex==3.19.0', 'azure-mgmt-signalr==1.2.0', 'sqlparse==0.4.3', 'parse-type==0.6.0', 'envier==0.5.1', 'gevent==23.9.1', 'openpyxl==3.1.0', 'ordered-set==4.0.2', 'openai==1.13.4', 'flask-sqlalchemy==3.1.0', 'fiona==1.9.6', 'selenium==4.18.1', 'rich==13.6.0', 'jupyterlab==4.1.5', 'cachetools==5.3.2', 'deprecation==2.1.0', 'boto3-stubs==1.34.63', 'stevedore==5.1.0', 'apache-airflow-providers-amazon==8.17.0', 'aws-xray-sdk==2.12.1', 'geoip2==4.7.0', 'executing==1.1.1', 'pre-commit==3.6.2', 'scp==0.14.4', 'querystring-parser==1.2.2', 'ruamel-yaml==0.18.5', 'aws-requests-auth==0.4.2', 'ipaddress==1.0.23', 'pre-commit==3.6.1', 'slack-sdk==3.27.0', 'html5lib==1.1', 'azure-mgmt-iotcentral==4.0.0', 'comm==0.2.1', 'google-pasta==0.1.8', 'jax==0.4.25', 'argon2-cffi==23.1.0', 'mmh3==4.0.0', 'cachecontrol==0.13.1', 'keras==3.0.4', 'setuptools-scm==8.0.2', 'fire==0.4.0', 'python-http-client==3.3.6', 'tensorboard-plugin-wit==1.8.1', 'pytest==8.1.1', 'pypdf==4.0.2', 'fonttools==4.48.1', 'pycares==4.2.2', 'pynacl==1.4.0', 'confection==0.1.2', 'feedparser==6.0.11', 'bleach==6.0.0', 'locket==0.2.1', 'tensorflow-io==0.35.0', 'grpcio-tools==1.62.0', 'aiohttp==3.9.1', 'aws-sam-translator==1.85.0', 'moto==5.0.1', 'datasets==2.18.0', 'google-cloud-logging==3.9.0', 'billiard==4.0.2', 'transformers==4.38.2', 'azure-mgmt-kusto==3.1.0', 'nvidia-nvjitlink-cu12==12.3.101', 'frozenlist==1.3.3', 'aioitertools==0.9.0', 'flask-login==0.6.2', 'docker==6.1.2', 'h3==3.7.7', 'zstandard==0.21.0', 'flask-jwt-extended==4.5.3', 'google-cloud-tasks==2.16.2', 'twine==4.0.2', 'opencensus-ext-azure==1.1.11', 'tomli==1.2.3', 'types-urllib3==1.26.25.14', 'py==1.10.0', 'requests-toolbelt==1.0.0', 'parsedatetime==2.5', 'kafka-python==2.0.2', 'aiosignal==1.3.1', 'azure-mgmt-batchai==2.0.0', 'pycodestyle==2.11.1', 'shapely==2.0.1', 'wsproto==1.0.0', 'pytest-metadata==3.1.1', 'fastapi==0.109.2', 'korean-lunar-calendar==0.2.1', 'azure-mgmt-containerregistry==10.1.0', 'aws-requests-auth==0.4.3', 'google-cloud-logging==3.10.0', 'lightning-utilities==0.10.1', 'wasabi==1.1.2', 'fuzzywuzzy==0.18.0', 'connexion==3.0.6', 'tensorflow-datasets==4.9.3', 'google-auth==2.28.1', 'psycopg==3.1.17', 'statsmodels==0.14.0', 'scikit-learn==1.4.0', 'langdetect==1.0.9', 'google-cloud-aiplatform==1.44.0', 'sh==2.0.4', 'langcodes==3.2.1', 'kombu==5.3.3', 'python-levenshtein==0.23.0', 'newrelic==9.6.0', 'jupyter-events==0.9.0', 'javaproperties==0.7.0', 'google-cloud-dataproc==5.9.3', 'pyodbc==5.0.1', 'mypy-boto3-redshift-data==1.34.0', 'lazy-object-proxy==1.8.0', 'fonttools==4.50.0', 'google-cloud-videointelligence==2.13.3', 'avro==1.11.3', 'linkify-it-py==2.0.3', 'packaging==23.2', 'asgiref==3.7.2', 'kfp-server-api==2.0.3', 'stack-data==0.6.1', 'progressbar2==4.4.0', 'google-cloud-resource-manager==1.12.3', 'pep517==0.13.0', 'hvac==2.0.0', 'google-re2==1.1', 'tzlocal==5.1', 'tomlkit==0.12.4', 'google-cloud-appengine-logging==1.4.1', 'pytest-html==4.1.1', 'uritemplate==4.0.0', 'types-pytz==2023.4.0.20240130', 'telethon==1.33.1', 'azure-mgmt-cognitiveservices==13.3.0', 'mistune==3.0.2', 'pycountry==23.12.7', 'tensorflow-probability==0.23.0', 'azure-mgmt-servicebus==8.1.0', 'keras-preprocessing==1.1.1', 'azure-mgmt-media==10.0.0', 'python-multipart==0.0.8', 'spacy==3.7.2', 'universal-pathlib==0.2.1', 'kubernetes==28.1.0', 'msrest==0.6.20', 'geopy==2.4.0', 'annotated-types==0.6.0', 'limits==3.9.0', 'smart-open==7.0.1', 'protobuf3-to-dict==0.1.5', 'dateparser==1.1.8', 'pyparsing==3.1.2', 'ldap3==2.9.1', 'azure-mgmt-policyinsights==0.5.0', 'kubernetes==29.0.0', 'secretstorage==3.3.2', 'azure-mgmt-advisor==3.0.0', 'installer==0.5.1', 'apache-airflow-providers-snowflake==5.3.0', 'azure-mgmt-storage==21.1.0', 'pyserial==3.4', 'fqdn==1.5.0', 'maxminddb==2.5.1', 'python-gitlab==4.2.0', 'hologram==0.0.16', 'opentelemetry-exporter-otlp-proto-common==1.22.0', 'lxml==5.0.0', 'pysocks==1.7.0', 'cryptography==42.0.3', 'lightning-utilities==0.9.0', 'mypy-boto3-s3==1.34.64', 'types-setuptools==69.2.0.20240316', 'service-identity==24.1.0', 'pylint==3.0.4', 'ml-dtypes==0.3.0', 'ijson==3.2.2', 'azure-core==1.29.7', 'azure-keyvault-keys==4.7.0', 'configargparse==1.5.3', 'google-cloud-compute==1.16.1', 'azure-mgmt-reservations==2.3.0', 'threadpoolctl==3.2.0', 'azure-mgmt-web==7.0.0', 'vine==5.0.0', 'pybind11==2.11.0', 'types-protobuf==4.24.0.20240311', 'azure-cosmos==4.6.0', 'toml==0.10.2', 'google-api-core==2.17.0', 'antlr4-python3-runtime==4.13.0', 'incremental==21.3.0', 'nbclient==0.10.0', 'numba==0.59.0', 'azure-mgmt-authorization==2.0.0', 'marshmallow==3.21.0', 'moto==5.0.2', 'jsonschema-specifications==2023.11.1', 'grpc-google-iam-v1==0.13.0', 'rpds-py==0.17.1', 'notebook-shim==0.2.2', 'opencensus-ext-azure==1.1.12', 'python-http-client==3.3.5', 'ddtrace==2.7.3', 'google-cloud-bigtable==2.21.0', 'hologram==0.0.15', 'backoff==2.2.0', 'llvmlite==0.42.0', 'apscheduler==3.10.2', 'h11==0.14.0', 'opencv-python==4.8.1.78', 'gsutil==5.27', 'tokenizers==0.15.2', 'flake8==6.0.0', 'pendulum==2.1.1', 'preshed==3.0.9', 'accelerate==0.27.1', 'pyotp==2.9.0', 'sshtunnel==0.3.1', 'sphinxcontrib-htmlhelp==2.0.5', 'firebase-admin==6.5.0', 'evergreen-py==3.6.23', 'google-cloud-monitoring==2.19.2', 'contextlib2==21.6.0', 'qtpy==2.3.1', 'smdebug-rulesconfig==0.1.7', 'fasteners==0.19', 'boltons==23.0.0', 'graphql-core==3.2.1', 'checkov==3.2.36', 'pynacl==1.3.0', 'send2trash==1.8.2', 'confection==0.1.4', 'mmh3==4.0.1', 'rich-argparse==1.3.0', 'pickleshare==0.7.5', 'scramp==1.4.4', 'google-re2==1.0', 'scramp==1.4.3', 'unidecode==1.3.7', 'decorator==5.0.9', 'parameterized==0.7.5', 'google-cloud-container==2.41.0', 'pypandoc==1.12', 'h5py==3.10.0', 'firebase-admin==6.3.0', 'voluptuous==0.13.1', 'cssselect==1.1.0', 'azure-mgmt-datalake-analytics==0.4.0', 'avro-python3==1.10.0', 'sqlalchemy-jsonfield==1.0.0', 'azure-multiapi-storage==1.2.0', 'certifi==2024.2.2', 'google-cloud-aiplatform==1.43.0', 'azure-mgmt-signalr==1.0.0', 'webcolors==1.13', 'click-repl==0.1.6', 'google-cloud-build==3.23.2', 'dulwich==0.21.7', 'pyflakes==3.1.0', 'django-filter==23.5', 'dataclasses-json==0.6.4', 'datetime==5.2', 'googleapis-common-protos==1.63.0', 'diskcache==5.6.1', 'azure-mgmt-redhatopenshift==1.4.0', 'pydash==7.0.5', 'keras-applications==1.0.8', 'requests-oauthlib==1.3.1', 'kr8s==0.14.0', 'azure-mgmt-msi==6.1.0', 'boto==2.47.0', 'requests==2.31.0', 'imbalanced-learn==0.10.1', 'azure-keyvault==4.0.0', 'lightgbm==4.2.0', 'portalocker==2.8.2', 'pyhive==0.6.4', 'xxhash==3.2.0', 'google-cloud-bigquery-storage==2.23.0', 'hdfs==2.7.1', 'gitpython==3.1.40', 'twine==4.0.1', 'types-redis==4.6.0.20240218', 'ptyprocess==0.6.0', 'types-protobuf==4.24.0.20240129', 'terminado==0.18.0', 'poetry==1.8.0', 'gevent==23.9.0', 'cinemagoer==2023.5.1', 'ndg-httpsclient==0.4.4', 'email-validator==2.1.1', 'authlib==1.2.0', 'tomli==2.0.1', 'fonttools==4.49.0', 'gunicorn==21.1.0', 'freezegun==1.3.0', 'patsy==0.5.6', 'uvicorn==0.27.0', 'leather==0.4.0', 'widgetsnbextension==4.0.9', 'pydata-google-auth==1.8.0', 'azure-mgmt-monitor==6.0.0', 'ecdsa==0.17.0', 'sphinx==7.2.4', 'dvclive==3.44.0', 'pydantic==2.6.3', 'fastparquet==2023.10.0', 'opentelemetry-exporter-otlp-proto-http==1.22.0', 'responses==0.24.1', 'azure-kusto-ingest==4.3.0', 'ldap3==2.9', 'defusedxml==0.7.1', 'azure-synapse-accesscontrol==0.6.0', 'cerberus==1.3.4', 'databricks-api==0.8.0', 'patsy==0.5.3', 'azure-mgmt-cdn==13.0.0', 'rapidfuzz==3.6.2', 'aiofiles==22.1.0', 'orbax-checkpoint==0.5.5', 'pytest-cov==3.0.0', 'pysftp==0.2.9', 'portalocker==2.8.1', 'requests-aws4auth==1.2.3', 'python-gnupg==0.5.1', 'bandit==1.7.8', 'geopandas==0.14.2', 'keras-preprocessing==1.1.2', 'gitdb==4.0.10', 'ultralytics==8.1.29', 'gcsfs==2024.2.0', 'great-expectations==0.18.9', 'azure-mgmt-datamigration==10.0.0', 'langcodes==3.2.0', 'openpyxl==3.1.2', 'google-cloud-dataproc==5.9.1', 'pickleshare==0.7.4', 'flask-babel==3.1.0', 'toolz==0.12.1', 'xmltodict==0.13.0', 'pycryptodomex==3.20.0', 'asttokens==2.3.0', 'oauthlib==3.2.2', 'ipykernel==6.29.2', 'logbook==1.7.0', 'pathy==0.11.0', 'nh3==0.2.15', 'parso==0.8.3', 'spark-nlp==5.2.3', 'ninja==1.10.2.4', 'sqlalchemy-utils==0.41.1', 'loguru==0.7.2', 'ray==2.9.3', 'rfc3339-validator==0.1.2', 'sqlalchemy==2.0.28', 'sshtunnel==0.3.2', 'mkdocs-material==9.5.12', 'google-cloud-bigquery==3.17.2', 'azure-eventhub==5.11.6', 'httpcore==1.0.3', 'cfn-lint==0.85.2', 'azure-mgmt-containerinstance==9.2.0', 'cryptography==42.0.4', 'msal-extensions==1.1.0', 'azure-mgmt-network==25.1.0', 'google-cloud-audit-log==0.2.3', 'azure-mgmt-datalake-analytics==0.6.0', 'proto-plus==1.23.0', 'zipp==3.18.0', 'limits==3.8.0', 'javaproperties==0.8.0', 'nvidia-cuda-runtime-cu12==12.3.52', 'python-jsonpath==1.1.1', 'cfgv==3.4.0', 'bytecode==0.15.1', 'google-resumable-media==2.7.0', 'reportlab==4.0.8', 'pygments==2.17.0', 'sphinxcontrib-htmlhelp==2.0.4', 'pycodestyle==2.11.0', 'hvac==2.1.0', 'tensorflow-io==0.34.0', 'cinemagoer==2022.12.27', 'pgpy==0.5.3', 'azure-synapse-artifacts==0.18.0', 'wcwidth==0.2.12', 'python-dateutil==2.9.0', 'azure-mgmt-cdn==12.0.0', 'gql==3.5.0', 'apache-airflow-providers-cncf-kubernetes==8.0.0', 'user-agents==2.1', 'google-cloud-core==2.4.1', 'jupyter-console==6.6.2', 'black==24.3.0', 'html5lib==0.999999999', 'uamqp==1.6.7', 'executing==1.2.0', 'httptools==0.5.0', 'natsort==8.4.0', 'sphinx==7.2.6', 'preshed==3.0.8', 'nvidia-cufft-cu12==11.0.11.19', 'azure-mgmt-advisor==4.0.0', 'xlsxwriter==3.2.0', 'thinc==8.2.1', 'cleo==2.0.1', 'matplotlib-inline==0.1.6', 'azure-keyvault-keys==4.8.0', 'pyarrow==15.0.1', 'pymssql==2.2.11', 'datadog==0.47.0', 'botocore-stubs==1.34.64', 'dill==0.3.6', 'sentry-sdk==1.40.6', 'snowballstemmer==2.0.0', 'altair==5.1.2', 'async-timeout==4.0.1', 'responses==0.25.0', 'phonenumbers==8.13.31', 'rpds-py==0.16.2', 'tensorflow-serving-api==2.13.1', 'azure-keyvault==4.2.0', 'botocore==1.34.63', 'gensim==4.3.2', 'httpx==0.26.0', 'gitdb==4.0.11', 'docker-pycreds==0.2.3', 'prison==0.2.0', 'azure-mgmt-cosmosdb==9.3.0', 'ordered-set==4.1.0', 'google-api-core==2.16.2', 'pytimeparse==1.1.8', 'nest-asyncio==1.5.8', 'sphinxcontrib-serializinghtml==1.1.8', 'pytest-asyncio==0.23.5', 'entrypoints==0.3', 'mistune==3.0.0', 'requests-file==2.0.0', 'google-pasta==0.2.0', 'lockfile==0.12.2', 'nvidia-cudnn-cu12==8.9.5.30', 'streamlit==1.32.1', 'aiofiles==23.2.1', 'uri-template==1.3.0', 'dpath==2.1.6', 'thrift==0.15.0', 'pymysql==1.1.0', 'azure-mgmt-dns==8.1.0', 'cloudpickle==2.2.0', 'py-cpuinfo==7.0.0', 'tensorflow-hub==0.15.0', 'tzdata==2024.1', 'humanize==4.9.0', 'stack-data==0.6.2', 'setuptools-scm==8.0.3', 'tox==4.13.0', 'flask-appbuilder==4.3.11', 'asyncache==0.2.0', 'tornado==6.4', 'flask-appbuilder==4.4.0', 'boltons==23.1.0', 'pycodestyle==2.10.0', 'progressbar2==4.4.1', 'google-ads==23.1.0', 'jeepney==0.7.1', 'mypy-extensions==1.0.0', 'torchvision==0.16.2', 'ruamel-yaml==0.18.3', 'distro==1.8.0', 'docker-pycreds==0.4.0', 'ec2-metadata==2.11.0', 'annotated-types==0.4.0', 'apache-airflow-providers-sqlite==3.7.0', 'google-cloud-tasks==2.16.3', 'azure-kusto-data==4.3.1', 'zope-interface==6.1', 'elastic-transport==8.10.0', 'awswrangler==3.6.0', 'typing==3.7.4', 'jsonlines==4.0.0', 'celery==5.3.4', 'python-editor==1.0.1', 'humanfriendly==9.2', 'azure-mgmt-iothub==2.4.0', 'azure-storage-file-datalake==12.14.0', 'qtpy==2.4.0', 'databricks-sql-connector==3.0.3', 'azure-storage-file-share==12.14.1', 'nvidia-cusolver-cu12==11.5.3.52', 'psycopg2-binary==2.9.8', 'azure-storage-queue==12.8.0', 'tldextract==5.0.1', 'azure-mgmt-network==25.2.0', 'mpmath==1.3.0', 'charset-normalizer==3.3.1', 'editables==0.5', 'azure-mgmt-apimanagement==2.1.0', 'azure-mgmt-imagebuilder==1.3.0', 'hyperframe==5.2.0', 'looker-sdk==24.0.0', 'jsondiff==1.3.1', 'pox==0.3.3', 'jedi==0.19.1', 'jupyter-client==8.6.1', 'jax==0.4.23', 'identify==2.5.33', 'partd==1.4.1', 'python-box==7.1.1', 'bcrypt==4.0.1', 'setuptools-scm==8.0.4', 'openai==1.14.1', 'parsedatetime==2.6', 'pandocfilters==1.5.1', 'more-itertools==10.1.0', 'enum34==1.1.9', 'azure-mgmt-redhatopenshift==1.2.0', 'matplotlib==3.8.3', 'poetry==1.8.1', 'azure-mgmt-hdinsight==9.0.0', 'pymsteams==0.2.2', 'azure-mgmt-policyinsights==1.0.0', 'gremlinpython==3.7.1', 'shap==0.44.0', 'xlwt==1.3.0', 'python-utils==3.8.0', 'protobuf==4.25.2', 'h11==0.13.0', 'stringcase==1.0.4', 'connexion==3.0.5', 'elasticsearch-dsl==8.12.0', 'azure-mgmt-cognitiveservices==13.5.0', 'pysocks==1.6.8', 'cssselect==1.2.0', 'asyncio==3.4.2', 'pbr==6.0.0', 'flake8==6.1.0', 'pypdf==4.1.0', 'coverage==7.4.4', 'dask==2024.3.1', 'huggingface-hub==0.21.2', 'pillow==10.2.0', 'databricks-sql-connector==3.0.2', 'azure-cli-core==2.57.0', 'qtpy==2.4.1', 'voluptuous==0.14.1', 'async-timeout==4.0.2', 'mccabe==0.6.0', 'django==4.2.10', 'google-cloud-resource-manager==1.12.1', 'python-gnupg==0.5.2', 'parso==0.8.2', 'json5==0.9.24', 'pyarrow==15.0.0', 'google-api-python-client==2.121.0', 'pathlib2==2.3.7', 'slackclient==2.9.2', 'xmltodict==0.11.0', 'pydeequ==1.1.1', 'types-python-dateutil==2.8.19.20240311', 'opentelemetry-proto==1.22.0', 'leather==0.3.3', 'mergedeep==1.3.2', 'distro==1.9.0', 'fastjsonschema==2.19.1', 'asttokens==2.4.0', 'polars==0.20.14', 'factory-boy==3.2.1', 'docopt==0.6.0', 'oscrypto==1.2.0', 'tqdm==4.66.0', 'requests-oauthlib==1.3.0', 'azure-identity==1.14.1', 'astroid==3.0.3', 'kornia==0.7.0', 'importlib-resources==6.3.1', 'tensorflow-hub==0.16.0', 'yarl==1.9.4', 'pytest==8.0.2', 'avro-python3==1.10.1', 'asyncache==0.3.0', 'torchmetrics==1.2.1', 'pymongo==4.6.0', 'types-protobuf==4.24.0.20240302', 'types-requests==2.31.0.20240310', 'cattrs==23.2.1', 'starlette==0.37.0', 'platformdirs==4.1.0', 'google-cloud-secret-manager==2.18.2', 'resolvelib==1.0.1', 'inflect==6.1.1', 'tinycss2==1.2.0', 'qtconsole==5.5.0', 'dbt-postgres==1.7.8', 'cligj==0.7.2', 'tensorboard==2.16.1', 'ftfy==6.2.0', 'dm-tree==0.1.7', 'pillow==10.1.0', 'elasticsearch-dsl==8.9.0', 'streamlit==1.32.0', 'jira==3.5.1', 'babel==2.14.0', 'pre-commit==3.6.0', 'filelock==3.12.4', 'google-cloud-pubsub==2.20.2', 'h2==4.0.0', 'confection==0.1.3', 'numba==0.58.0', 'smart-open==7.0.0', 'aenum==3.1.14', 'azure-mgmt-hdinsight==7.0.0', 'wandb==0.16.4', 'junit-xml==1.7', 'wcwidth==0.2.13', 'envier==0.5.0', 'uamqp==1.6.6', 'mypy-boto3-rds==1.34.57', 'statsmodels==0.13.5', 'oscrypto==1.2.1', 'polars==0.20.15', 'torchmetrics==1.3.0', 'sendgrid==6.9.7', 'azure-mgmt-rdbms==10.0.0', 'ua-parser==0.16.1', 'marshmallow-sqlalchemy==0.30.0', 'python-daemon==2.3.2', 'poetry-core==1.8.1', 'junit-xml==1.8', 'pyproject-api==1.5.4', 'pytest-timeout==2.2.0', 'azure-keyvault-secrets==4.6.0', 'contourpy==1.1.0', 'pathspec==0.12.1', 'azure-mgmt-redhatopenshift==1.3.0', 'azure-mgmt-core==1.3.2', 'langsmith==0.1.26', 'azure-mgmt-advisor==9.0.0', 'attrs==22.2.0', 'azure-mgmt-botservice==0.3.0', 'azure-mgmt-iothubprovisioningservices==1.1.0', 'googleapis-common-protos==1.62.0', 'prettytable==3.10.0', 'boto3==1.34.63', 'opencensus==0.11.2', 'cligj==0.7.0', 'slack-sdk==3.26.2', 'gensim==4.3.0', 'flask-appbuilder==4.4.1', 'azure-mgmt-datafactory==4.0.0', 'pathlib==1.0.1', 'structlog==24.1.0', 'cloudpickle==2.2.1', 'pep517==0.12.0', 'rapidfuzz==3.6.1', 'tensorstore==0.1.54', 'xarray==2024.2.0', 'opentelemetry-exporter-otlp==1.23.0', 'connexion==3.0.4', 'jupyter-core==5.7.1', 'ratelimit==2.1.0', 'jsonpatch==1.32', 'tensorflow-text==2.15.0', 'types-awscrt==0.20.3', 'azure-mgmt-containerregistry==10.3.0', 'marshmallow-sqlalchemy==1.0.0', 'numexpr==2.8.7', 'aws-requests-auth==0.4.1', 'tensorflow-io-gcs-filesystem==0.34.0', 'opentelemetry-exporter-otlp-proto-grpc==1.22.0', 'prometheus-client==0.18.0', 'types-urllib3==1.26.25.12', 'smart-open==6.4.0', 'python-dotenv==0.21.1', 'websocket-client==1.7.0', 'botocore-stubs==1.34.62', 'telethon==1.33.0', 'azure-mgmt-compute==30.5.0', 'cfgv==3.3.0', 'asynctest==0.12.3', 'sympy==1.11.1', 'yamllint==1.35.1', 'pymysql==1.0.3', 'jaydebeapi==1.2.1', 'gremlinpython==3.6.6', 'parameterized==0.9.0', 'pathspec==0.11.2', 'nvidia-cusparse-cu12==12.1.3.153', 'markupsafe==2.1.4', 'pathos==0.3.1', 'azure-appconfiguration==1.5.0', 'mdit-py-plugins==0.3.4', 'sphinx-rtd-theme==1.3.0', 'cdk-nag==2.28.64', 'certifi==2023.7.22', 'xlsxwriter==3.1.8', 'google-cloud-compute==1.17.0', 'pendulum==2.1.2', 'oauth2client==4.1.3', 'evaluate==0.4.1', 'scandir==1.10.0', 'click-repl==0.2.0', 'pandas==2.2.0', 'fastapi==0.110.0', 'dbt-extractor==0.5.1', 'apache-beam==2.54.0', 'wtforms==3.1.2', 'installer==0.7.0', 'pytz==2023.3', 'ray==2.9.2', 'pytest-forked==1.5.0', 'scikit-image==0.20.0', 'jinja2==3.1.3', 'amqp==5.2.0', 'ruamel-yaml-clib==0.2.6', 'sortedcontainers==2.3.0', 'cython==3.0.7', 'jira==3.6.0', 'google-cloud-storage==2.15.0', 'semantic-version==2.9.0', 'configupdater==3.2', 'python-daemon==3.0.1', 'mypy-boto3-s3==1.34.62', 'uvicorn==0.27.1', 'azure-mgmt-resource==23.0.0', 'types-setuptools==69.1.0.20240310', 'pure-eval==0.2.1', 'google==2.0.2', 'azure-mgmt-security==6.0.0', 'terminado==0.18.1', 'scipy==1.11.4', 'soupsieve==2.5', 'pyserial==3.5', 'azure-mgmt-keyvault==10.2.2', 'starlette==0.37.1', 'azure-mgmt-trafficmanager==1.1.0', 'attrs==23.1.0', 'inflect==7.0.0', 'unittest-xml-reporting==3.1.0', 'virtualenv==20.25.1', 'notebook==7.1.0', 'aws-sam-translator==1.86.0', 'networkx==3.2', 'junit-xml==1.9', 'azure-mgmt-resource==22.0.0', 'kfp==2.5.0', 'grpcio-status==1.60.1', 'google-cloud-datacatalog==3.18.2', 'azure-mgmt-dns==3.0.0', 'apache-beam==2.52.0', 'dill==0.3.8', 'xxhash==3.4.1', 'alembic==1.12.1', 'iniconfig==1.1.1', 'nvidia-cudnn-cu12==8.9.7.29', 'jmespath==1.0.0', 'multidict==6.0.4', 'psutil==5.9.6', 'pypandoc==1.11', 'google-ads==23.0.0', 'rich-argparse==1.2.0', 'passlib==1.7.4', 'databricks-sdk==0.22.0', 'pandas-gbq==0.22.0', 'langdetect==1.0.7', 'hiredis==2.2.2', 'ddsketch==2.0.4', 'async-generator==1.8', 'python-jose==3.1.0', 'croniter==2.0.1', 'pypdf==4.0.1', 'pandocfilters==1.5.0', 'deprecation==2.0.7', 'pytest-metadata==3.1.0', 'mysql-connector-python==8.3.0', 'gradio==4.20.1', 'dataclasses==0.6', 'tensorflow-io-gcs-filesystem==0.36.0', 'tqdm==4.66.1', 'faker==24.1.0', 'readme-renderer==42.0', 'azure-mgmt-synapse==0.8.0', 'distributed==2024.2.1', 'flask-login==0.6.1', 'boto3-stubs==1.34.64', 'networkx==3.1', 'typing-inspect==0.9.0', 'jsonpointer==2.2', 'scikit-image==0.22.0', 'delta-spark==3.1.0', 'keyring==24.3.1', 'einops==0.6.0', 'bracex==2.4', 'tensorboard-data-server==0.7.2', 'funcsigs==1.0.0', 'xyzservices==2023.10.0', 'azure-synapse-accesscontrol==0.7.0', 'azure-mgmt-sqlvirtualmachine==0.5.0', 'bleach==5.0.1', 'requests==2.30.0', 'nodeenv==1.7.0', 'sphinx-rtd-theme==2.0.0', 'hpack==4.0.0', 'monotonic==1.6', 'fsspec==2023.12.2', 'sniffio==1.3.0', 'werkzeug==3.0.0', 'agate==1.9.1', 'ultralytics==8.1.28', 'azure-mgmt-network==25.3.0', 'marshmallow==3.20.2', 'azure-mgmt-redis==14.1.0', 'tensorflow==2.15.0', 'grpcio-status==1.62.1', 'tenacity==8.2.3', 'pydantic-core==2.16.1', 'astor==0.8.0', 'azure-mgmt-sqlvirtualmachine==0.3.0', 'ratelimit==2.2.1', 'xgboost==2.0.2', 'httpx==0.27.0', 'altair==5.1.1', 'bytecode==0.14.2', 'py-cpuinfo==8.0.0', 'pandas-gbq==0.20.0', 'azure-cli-core==2.58.0', 'fabric==3.2.1', 'jpype1==1.4.0', 'apache-airflow-providers-cncf-kubernetes==7.14.0', 'flask-limiter==3.5.1', 'flask-sqlalchemy==3.0.5', 'tensorflow-text==2.14.0', 'mypy==1.8.0', 'opentelemetry-exporter-otlp-proto-http==1.21.0', 'rich-argparse==1.4.0', 'pexpect==4.8.0', 'pyhive==0.6.5', 'marshmallow-dataclass==8.5.13', 'astunparse==1.6.1', 'apache-airflow-providers-snowflake==5.3.1', 'ply==3.9', 'nvidia-cusolver-cu12==11.6.0.99', 'pyperclip==1.8.2', 'wtforms==3.1.1', 'installer==0.6.0', 'uvloop==0.19.0', 'azure-mgmt-web==7.2.0', 'cymem==2.0.8', 'billiard==4.1.0', 'scandir==1.9.0', 'websocket-client==1.6.4', 'aws-sam-translator==1.84.0', 'tinycss2==1.1.1', 'graphviz==0.20.1', 'azure-kusto-ingest==4.2.0', 'google-cloud-datastore==2.19.0', 'google-cloud-container==2.43.0', 'setuptools-rust==1.8.0', 'comm==0.2.2', 'debugpy==1.7.0', 'python-jsonpath==1.0.0', 'grpc-google-iam-v1==0.12.7', 'orjson==3.9.15', 'natsort==8.3.1', 'apispec==6.5.0', 'lxml==5.0.1', 'deepdiff==6.7.0', 'responses==0.24.0', 'jsonschema==4.21.1', 'oauth2client==4.1.1', 'mergedeep==1.3.4', 'types-awscrt==0.20.5', 'proto-plus==1.22.2', 'hatchling==1.21.0', 'makefun==1.15.2', 'google-cloud-spanner==3.44.0', 'pendulum==3.0.0', 'cattrs==23.2.2', 'graphql-core==3.2.3', 'chex==0.1.83', 'dataclasses==0.5', 'portalocker==2.7.0', 'loguru==0.7.1', 'rdflib==6.3.2', 'arrow==1.2.3', 'distributed==2024.3.1', 'azure-mgmt-authorization==4.0.0', 'nltk==3.8', 'terminado==0.17.1', 'aiobotocore==2.12.1', 'authlib==1.3.0', 'fiona==1.9.4', 'email-validator==2.0.0', 'xgboost==2.0.1', 'overrides==7.5.0', 'torch==2.1.2', 'google-cloud-dlp==3.15.3', 'email-validator==2.1.0', 'gitpython==3.1.41', 'anyio==4.3.0', 'envier==0.4.0', 'mistune==3.0.1', 'nvidia-nccl-cu12==2.20.5', 'jedi==0.19.0', 'imdbpy==2021.4.18', 'numpy==1.26.4', 'jupyterlab-pygments==0.2.2', 'flask-limiter==3.5.0', 'azure-mgmt-datamigration==9.0.0', 'thinc==8.2.3', 'flask-wtf==1.2.1', 'typer==0.7.0', 'jupyter-console==6.6.3', 'coloredlogs==15.0.1', 'types-s3transfer==0.10.0', 'flask-wtf==1.1.2', 'inflection==0.4.0', 'tox==4.14.0', 'srsly==2.4.6', 'requests-mock==1.9.3', 'parse==1.20.1', 'statsd==4.0.1', 'torchvision==0.17.0', 'jsonpointer==2.3', 'identify==2.5.35', 'docker==7.0.0', 'h3==3.7.6', 'requests-oauthlib==1.4.0', 'pymeeus==0.5.12', 'funcsigs==1.0.2', 'safetensors==0.4.1', 'azure-cli-core==2.56.0', 'cron-descriptor==1.4.0', 'deprecated==1.2.14', 'azure-mgmt-iotcentral==9.0.0', 'click-didyoumean==0.3.0', 'aiobotocore==2.12.0', 'mypy-boto3-appflow==1.33.0', 'watchdog==4.0.0', 'h5py==3.9.0', 'multiprocess==0.70.15', 'sentry-sdk==1.41.0', 'fqdn==1.5.1', 'typed-ast==1.5.5', 'configupdater==3.1.1', 'opensearch-py==2.4.0', 'azure-mgmt-sql==3.0.1', 'ratelimit==2.2.0', 'tensorflow-io==0.36.0', 'awswrangler==3.7.1', 'pypandoc==1.13', 'kornia==0.7.1', 'aiohttp==3.9.3', 'nodeenv==1.6.0', 'google-cloud-resource-manager==1.12.2', 'azure-mgmt-devtestlabs==4.0.0', 'dvc-render==1.0.1', 'pynacl==1.5.0', 'constructs==10.2.69', 'xxhash==3.3.0', 'tzlocal==5.0.1', 'google-cloud-pubsub==2.20.0', 'openapi-spec-validator==0.7.0', 'scp==0.14.5', 'apache-airflow-providers-http==4.9.0', 'uritemplate==3.0.1', 'fastavro==1.9.2', 'tensorflow-metadata==1.13.0', 'pyserial==3.3', 'loguru==0.7.0', 'tifffile==2023.12.9', 'azure-identity==1.15.0', 'mysql-connector-python==8.1.0', 'text-unidecode==1.1', 'watchtower==3.0.1', 'pycparser==2.21', 'jupyter-console==6.6.1', 'tokenizers==0.15.0', 'huggingface-hub==0.21.4', 'importlib-metadata==7.0.1', 'configparser==6.0.0', 'ijson==3.2.1', 'geopy==2.3.0', 'tenacity==8.2.1', 'trio-websocket==0.10.3', 'distrax==0.1.4', 'dask==2024.3.0', 'cachecontrol==0.12.14', 'configargparse==1.7', 'flax==0.8.2', 'ray==2.9.1', 'pandas-gbq==0.21.0', 'databricks-api==0.7.0', 'diskcache==5.6.3', 'libclang==16.0.0', 'prompt-toolkit==3.0.43', 'webdriver-manager==4.0.1', 'oscrypto==1.3.0', 'opencensus==0.11.3', 'requests-file==1.5.0', 'nbconvert==7.16.0', 'python-jsonpath==1.1.0', 'execnet==2.0.1', 'debugpy==1.8.1', 'webencodings==0.5.1', 'azure-mgmt-sqlvirtualmachine==0.4.0', 'databricks-cli==0.17.7', 'pywavelets==1.2.0', 'jaraco-classes==3.3.0', 'uc-micro-py==1.0.2', 'zeep==4.2.1', 'opentelemetry-sdk==1.21.0', 'ruamel-yaml-clib==0.2.8', 'aws-psycopg2==1.2.1', 'onnx==1.14.0', 'slicer==0.0.6', 'shapely==2.0.3', 'python-gnupg==0.5.0', 'nh3==0.2.13', 'rfc3339-validator==0.1.4', 'matplotlib==3.8.2', 'humanfriendly==9.1', 'pycares==4.3.0', 'aws-psycopg2==1.3.8', 'async-lru==2.0.3', 'lz4==4.3.3', 'idna==3.6', 'numpy==1.26.2', 'parameterized==0.8.1', 'torchmetrics==1.3.1', 'xlwt==1.1.2', 'alembic==1.13.1', 'pycryptodome==3.19.0', 'alabaster==0.7.16', 'flask-session==0.4.1', 'bandit==1.7.7', 'azure-batch==14.1.0', 'google-cloud-appengine-logging==1.4.3', 'azure-appconfiguration==1.3.0', 'pytorch-lightning==2.2.0', 'incremental==17.5.0', 'kornia==0.7.2', 'jmespath==1.0.1', 'einops==0.7.0', 'pytest-mock==3.12.0', 'pycares==4.4.0', 'murmurhash==1.0.8', 'orjson==3.9.14', 'multidict==6.0.3', 'retry==0.9.2', 'google-resumable-media==2.6.0', 'pyelftools==0.29', 'frozendict==2.4.0', 'frozendict==2.3.9', 'pytorch-lightning==2.2.1', 'pytest-html==4.0.2', 'oldest-supported-numpy==2023.10.25', 'trio-websocket==0.10.4', 'jaydebeapi==1.2.2', 'smmap==5.0.0', 'nvidia-cublas-cu12==12.4.2.65', 'jupyter-core==5.7.2', 'knack==0.10.0', 'mako==1.3.0', 'thrift==0.14.2', 'sshtunnel==0.4.0', 'sentry-sdk==1.42.0', 'cloudpathlib==0.18.1', 'tokenizers==0.15.1', 'trio==0.23.1', 'jupyterlab-server==2.25.4', 'flask==3.0.2', 'protobuf==4.25.3', 'geoip2==4.8.0', 'azure-mgmt-trafficmanager==0.51.0', 'mock==5.0.1', 'avro-python3==1.10.2', 'azure-kusto-data==4.2.0', 'ndg-httpsclient==0.5.0', 'azure-mgmt-core==1.4.0', 'markdown==3.5.1', 'networkx==3.2.1', 'langdetect==1.0.8', 'dbt-core==1.7.9', 'redshift-connector==2.0.918', 'tensorflow-io-gcs-filesystem==0.35.0', 'shellingham==1.5.4', 'msgpack==1.0.6', 'partd==1.3.0', 'pbr==5.11.0', 'jaxlib==0.4.24', 'mccabe==0.7.0', 'aniso8601==9.0.0', 'opt-einsum==3.2.0', 'azure-mgmt-eventhub==11.0.0', 'apache-airflow-providers-common-sql==1.10.1', 'nvidia-cublas-cu12==12.3.4.1', 'korean-lunar-calendar==0.3.1', 'tensorflow==2.16.1', 'cleo==2.1.0', 'tornado==6.3.3', 'azure-mgmt-datalake-analytics==0.5.0', 'colorlog==6.8.2', 'gspread==6.0.2', 'distlib==0.3.6', 'keras-applications==1.0.7', 'pydot==2.0.0', 'pymsteams==0.2.1', 'msgpack==1.0.7', 'mypy-boto3-s3==1.34.14', 'shellingham==1.5.2', 'wheel==0.41.3', 'mypy-extensions==0.4.4', 'werkzeug==2.3.8', 'azure-mgmt-datafactory==6.0.0', 'python-multipart==0.0.7', 'croniter==2.0.0', 'locket==1.0.0', 'pathlib==0.97', 'pylint==3.0.3', 'notebook==7.1.1', 'monotonic==1.5', 'markdown-it-py==2.2.0', 'flake8==7.0.0', 'prometheus-client==0.19.0', 'pygments==2.17.1', 'google-cloud-firestore==2.14.0', 'blis==0.9.1', 'tblib==3.0.0', 'google-cloud-dlp==3.15.2', 'dnspython==2.5.0', 'nbclient==0.9.1', 'moto==5.0.3', 'zope-interface==6.0', 'bitarray==2.9.1', 'lz4==4.3.1', 'matplotlib-inline==0.1.5', 'google-crc32c==1.2.0', 'msrestazure==0.6.2', 'google-cloud-compute==1.18.0', 'azure-mgmt-servicebus==8.0.0', 'bokeh==3.4.0', 'pyjwt==2.7.0', 'querystring-parser==1.2.4', 'flask-babel==4.0.0', 'db-dtypes==1.2.0', 'exceptiongroup==1.1.3', 'langcodes==3.3.0', 'reportlab==4.0.9', 'azure-mgmt-redis==14.2.0', 'tensorboard-plugin-wit==1.7.0', 'yapf==0.32.0', 'uvloop==0.18.0', 'outcome==1.2.0', 'pytest-mock==3.11.1', 'async-generator==1.9', 'ldap3==2.8.1', 'robotframework-seleniumlibrary==6.2.0', 'keras==3.0.3', 'azure-mgmt-marketplaceordering==1.0.0', 'hpack==2.3.0', 'mypy-boto3-redshift-data==1.29.0', 'colorama==0.4.4', 'azure-mgmt-batch==17.1.0', 'elastic-transport==8.11.0', 'configparser==5.3.0', 'uvicorn==0.28.0', 'google-cloud-build==3.23.1', 'prompt-toolkit==3.0.42', 'fiona==1.9.5', 'cachelib==0.10.2', 'tox==4.14.1', 'nvidia-curand-cu12==10.3.4.101', 'nbconvert==7.16.1', 'async-generator==1.10', 'ruff==0.3.2', 'jupyter-lsp==2.2.4', 'pexpect==4.9.0', 'cmdstanpy==1.1.0', 'cligj==0.7.1', 'python-docx==1.1.0', 'databricks-sdk==0.20.0', 'mlflow==2.10.2', 'jsonpickle==3.0.2', 's3transfer==0.10.1', 'tensorflow==2.15.1', 'cycler==0.12.0', 'opt-einsum==3.2.1', 'cython==3.0.8', 'faker==24.1.1', 'django-cors-headers==4.3.1', 'bleach==6.1.0', 'srsly==2.4.7', 'resolvelib==0.9.0', 'hiredis==2.3.2', 'slackclient==2.9.4', 'semver==3.0.0', 'cycler==0.11.0', 'marshmallow-dataclass==8.5.14', 'pytest-rerunfailures==13.0', 'evidently==0.4.16', 'geopy==2.4.1', 'contextlib2==0.6.0', 'nose==1.3.4', 'langchain==0.1.10', 'urllib3==2.1.0', 'mdit-py-plugins==0.4.0', 'catalogue==2.0.10', 'anyio==4.1.0', 'flask-sqlalchemy==3.1.1', 'tomlkit==0.12.3', 'cookiecutter==2.5.0', 'mkdocs-material==9.5.13', 'texttable==1.7.0', 'flask-caching==2.0.1', 'pyhcl==0.4.4', 'starkbank-ecdsa==2.0.3', 'sentence-transformers==2.5.1', 'geographiclib==1.50', 'apache-airflow-providers-sqlite==3.7.1', 'nvidia-cuda-runtime-cu12==12.3.101', 'pypdf2==2.12.1', 'orjson==3.9.13', 'twilio==8.13.0', 'cachelib==0.12.0', 'invoke==2.2.0', 'jupyter-server-terminals==0.5.3', 'pycountry==23.12.11', 'ppft==1.7.6.8', 'cmake==3.27.9', 'pydata-google-auth==1.8.2', 'joblib==1.3.1', 'configobj==5.0.8', 'requests-toolbelt==0.10.0', 'dvclive==3.45.0', 'sqlalchemy-jsonfield==0.9.0', 'pygithub==2.1.1', 'tableauserverclient==0.30', 'tblib==1.7.0', 'pydantic==2.6.2', 'typer==0.8.0', 'asyncache==0.3.1', 'azure-cli==2.58.0', 'altair==5.2.0', 'bcrypt==4.1.2', 'flit-core==3.8.0', 'geopandas==0.14.1', 'pyspark==3.5.0', 'jupyter-lsp==2.2.2', 'types-setuptools==69.1.0.20240309', 'nvidia-nvjitlink-cu12==12.3.52', 'h3==3.7.4', 'langsmith==0.1.27', 'nvidia-cusparse-cu12==12.2.0.103', 'pydash==7.0.6', 'apscheduler==3.10.3', 'itsdangerous==2.1.1', 'pyathena==3.4.0', 'azure-keyvault-certificates==4.6.0', 'azure-mgmt-dns==8.0.0', 'diskcache==5.6.0', 'azure-mgmt-eventgrid==10.1.0', 'platformdirs==4.0.0', 'google-auth-httplib2==0.2.0', 'requests-ntlm==1.0.0', 'datetime==5.4', 'simplejson==3.19.1', 'ppft==1.7.6.6', 'google-cloud-dataproc==5.9.2', 'scikit-image==0.21.0', 'pyaml==23.9.6', 'cx-oracle==8.2.1', 'pyee==11.1.0', 'uri-template==1.1.0', 'nltk==3.7', 'google-auth==2.28.0', 'contourpy==1.2.0', 'torchvision==0.17.1', 'tabulate==0.8.10', 'azure-mgmt-cdn==11.0.0', 'more-itertools==10.2.0', 'python-magic==0.4.25', 'pgpy==0.5.4', 'google-cloud-kms==2.21.3', 'nose==1.3.6', 'elasticsearch==8.12.0', 'docstring-parser==0.14.1', 'kfp-server-api==2.0.5', 'evergreen-py==3.6.22', 'statsd==4.0.0', 'cmdstanpy==1.2.1', 'botocore==1.34.64', 'traitlets==5.14.0', 'smdebug-rulesconfig==1.0.1', 'mypy-boto3-rds==1.34.63', 'kiwisolver==1.4.3', 'azure-data-tables==12.4.4', 'parse-type==0.6.1', 'google-auth-oauthlib==1.1.0', 'pyspnego==0.9.2', 'flask-login==0.6.3', 'google-cloud-language==2.13.1', 'pydeequ==1.1.0', 'werkzeug==3.0.1', 'tensorflow-estimator==2.15.0', 'pytest-xdist==3.3.1', 'aws-lambda-powertools==2.35.1', 'click-plugins==1.0.4', 'apache-airflow-providers-sqlite==3.6.0', 'arrow==1.3.0', 'catalogue==2.0.8', 'cog==0.9.3', 'google-cloud-vision==3.7.0', 'gradio-client==0.11.0', 'pathlib2==2.3.5', 'service-identity==23.1.0', 'sqlparse==0.4.4', 'argparse==1.2.2', 'opentelemetry-exporter-otlp==1.22.0', 'jsonlines==3.1.0', 'zstandard==0.22.0', 'patsy==0.5.4', 'ipykernel==6.29.3', 'jsonpath-ng==1.5.3', 'shapely==2.0.2', 'pydub==0.25.0', 'azure-mgmt-containerregistry==10.2.0', 'azure-mgmt-eventhub==10.0.0', 'pyzmq==25.1.2', 'kfp-pipeline-spec==0.2.1', 'azure-batch==13.0.0', 'sentencepiece==0.1.99', 'types-pyyaml==6.0.12.20240311', 'azure-mgmt-msi==6.0.1', 'flax==0.8.0', 'dask==2024.2.1', 'editables==0.3', 'azure-multiapi-storage==1.1.0', 'onnxruntime==1.17.1', 'twilio==9.0.2', 'appdirs==1.4.2', 'db-dtypes==1.1.1', 'dvclive==3.43.0', 'protobuf3-to-dict==0.1.3', 'shortuuid==1.0.11', 'setuptools==69.1.0', 'zeep==4.1.0', 'apache-airflow==2.8.3', 'rdflib==6.3.1', 'numpy==1.26.3', 'regex==2023.8.8', 'omegaconf==2.2.2', 'websockets==12.0', 'flask-cors==4.0.0', 'stringcase==1.2.0', 'imbalanced-learn==0.12.0', 'click-plugins==1.1', 'docker==6.1.3', 'decorator==5.1.1', 'markdown-it-py==2.1.0', 'google-cloud-spanner==3.42.0', 'boltons==23.1.1', 'msrestazure==0.6.3', 'iso8601==2.1.0', 'imbalanced-learn==0.11.0', 'pytest-forked==1.4.0', 'fastavro==1.9.3', 'opentelemetry-exporter-otlp-proto-grpc==1.23.0', 'sqlalchemy==2.0.27', 'absl-py==2.0.0', 'azure-mgmt-consumption==8.0.0', 'apache-airflow-providers-snowflake==5.2.1', 'cron-descriptor==1.3.0', 'prison==0.2.1', 'aiofiles==23.1.0', 'tensorflow-datasets==4.9.2', 'click-didyoumean==0.2.0', 'magicattr==0.1.4', 'azure-cosmos==4.5.0', 'coverage==7.4.3', 'fasteners==0.18', 'snowflake-connector-python==3.7.1', 'pydub==0.24.1', 'applicationinsights==0.11.8', 'pymsteams==0.1.16', 'setproctitle==1.3.1', 'azure-mgmt-compute==30.6.0', 'imageio==2.34.0', 'nbclient==0.9.0', 'google-cloud-bigtable==2.23.0', 'twine==5.0.0', 'sqlparse==0.4.2', 'distlib==0.3.7', 'nvidia-cudnn-cu11==8.9.5.30', 'multidict==6.0.5', 'trove-classifiers==2024.2.23', 'pyspnego==0.10.2', 'azure-storage-file-datalake==12.13.1', 'comm==0.2.0', 'gql==3.4.0', 'pathy==0.10.2', 'qrcode==7.4.1', 'chardet==5.0.0', 'importlib-resources==6.2.0', 'spacy-legacy==3.0.10', 'pyparsing==3.1.0', 'statsmodels==0.14.1', 'python-json-logger==2.0.6', 'azure-cli-telemetry==1.0.8', 'leather==0.3.4', 'kr8s==0.13.5', 'openapi-spec-validator==0.6.0', 'black==24.2.0', 'opencensus-ext-azure==1.1.13', 'oldest-supported-numpy==2023.12.12', 'pyodbc==5.0.0', 'entrypoints==0.2.3', 'botocore-stubs==1.34.63', 'typeguard==4.1.4', 'opencv-python==4.9.0.80', 'resolvelib==1.0.0', 'setuptools-rust==1.9.0', 'click==8.1.6']

    package_ver_d = defaultdict(list)
    for package in packages_l:
        package_l = package.split("==")
        package_ver_d[package_l[0]].append(package)
    packages_ll["data_4"]=list(package_ver_d.keys())
    package_ver_dd["data_4"] = package_ver_d
    n_samples_d["data_4"]=4
    test_portion_d["data_4"]=0.25
    # ###################### choose CV batch ######################
    for test_sample_batch_idx in [0]:   # this is the initial idx of a test batch, i.e., if test batch size is 4*0.25 = 1, `test_sample_batch_idx`` means pick the 0-index element as the test batch.
        # #############################################################

        for dataset in ["data_4"]:
            n_samples = n_samples_d[dataset]
            test_portion = test_portion_d[dataset]
            packages_l = packages_ll[dataset]
            for (with_filter, freq) in [[(False, 100),(True, 50),(True, 25),(True, 15)][0]]:  # [(False, 100), (True, 50), (True, 75), (True, 100)]
                if with_filter:
                    # Consider a set with tokens to filter
                    with open(f"/home/cc/Praxi-study/Praxi-Pipeline/data/{dataset}/filters/tagsets_SL_tagnames_reoccurentcount_d", 'rb') as tf:
                        tokens_filter_set = yaml.load(tf, Loader=yaml.Loader)
                else:
                    tokens_filter_set = set()
                for n_jobs in [32]:
                    for n_models, test_batch_count in zip([[1000,500,50,25,10,5,2,1][7]],[1,1,1,1,1,1,1,1,1,1]): # zip([1000,750,500,50,25,10,1],[1,1,1,1]):
                        for n_estimators in [100]:
                            for depth in [1]:
                                for tree_method in["exact"]: # "exact","approx","hist"
                                    for max_bin in [1]:
                                        for input_size, dim_compact_factor in zip([None],[1,1,1,1,1]): # [None, 6832, 13664, 27329, 54659,109319],[1,1,1,1] # [None, 500, 1000, 5000, 10000, 15000],[1,1,1,1,1,1]
                                            random_instance = random.Random(4)
                                            for shuffle_idx in range(0,10):
                                                # sample labels per sub-model
                                                randomized_packages_l = random_instance.sample(packages_l, len(packages_l))
                                                package_subset, step = [], len(randomized_packages_l)//n_models
                                                for i in range(0, len(randomized_packages_l), step):
                                                    if dataset in package_ver_dd:
                                                        a_subset = set()
                                                        for package in randomized_packages_l[i:i+step]:
                                                            a_subset.update([package.replace("==", "_v").replace(".","_") for package in package_ver_dd[dataset][package]])
                                                            
                                                    else:
                                                        a_subset = set(randomized_packages_l[i:i+step])
                                                    package_subset.append(a_subset)

                                                all_test_tag_files_l = [] # for testing SL data from trainset
                                                # cross-validation for samples
                                                for i, train_subset in enumerate(package_subset):
                                                    train_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/tagsets_SL/"
                                                    test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/tagsets_SL/" # Cross Validation: testing a portion of the SL dataset
                                                    
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
                                                    all_test_tag_files_l.extend(test_tag_files_l)
                                                    cwd  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak_detailed_encoder_times_32encoder/"
                                                    run_init_train(train_tags_path, test_tags_path, cwd, train_tags_init_l=train_tag_files_l, test_tags_l=test_tag_files_l, n_jobs=n_jobs, n_estimators=n_estimators, tokens_filter_set=tokens_filter_set, test_batch_count=test_batch_count, input_size=input_size, compact_factor=dim_compact_factor, depth=depth, tree_method=tree_method, freq=freq)
                                                    # break


        ###################################
        # run_iter_train()


        # # ###################################
        # # run_pred()
        # # Testing the ML dataset
        # for dataset in ["data_4"]:
        #     n_samples = n_samples_d[dataset]
        #     for (with_filter, freq) in [(False, 100)]:
        #         for n_jobs in [32]:
        #             for n_models, test_batch_count in zip([[50,25,10,1][3]],[1,1,1,1,1,1,1,1,1,1]): # zip([50,25,20,15,10,5,1],[1,1,1,1,1,1,8]):
        #                 for n_estimators in [100]:
        #                     for depth in [1]:
        #                         for tree_method in["exact"]: # "exact","approx","hist"
        #                             for max_bin in [1]:
        #                                 for input_size, dim_compact_factor in zip([None],[1,1,1,1,1]):
        #                                     for shuffle_idx in range(1):

                                                for clf_njobs in [32]:
                                                    clf_path = []
                                                    for i in range(n_models):
                                                        # clf_pathname = "/home/cc/test/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(clf_njobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak_detailed_encoder_times/model_init.json"
                                                        clf_pathname = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak_detailed_encoder_times_32encoder/model_init.json"
                                                        if os.path.isfile(clf_pathname):
                                                            clf_path.append(clf_pathname)
                                                        else:
                                                            print(f"clf is missing! {clf_pathname}")
                                                            sys.exit(-1)
                                                            # break
                                                    cwd = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak_detailed_encoder_times_32encoder/"
                                                    # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/big_ML_biased_test/"
                                                    test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/tagsets_ML/"
                                                    # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/tagsets_SL/"
                                    #    run_init_train(train_tags_path, test_tags_path, cwd, n_jobs=n_jobs, n_estimators=n_estimators, train_packages_select_set=train_subset, test_packages_select_set=test_subset, input_size=input_size, depth=depth, tree_method=tree_method)
                                                    run_pred(cwd, clf_path, test_tags_path, tag_files_l=all_test_tag_files_l, flag_load_obj=True, n_jobs=n_jobs, n_estimators=n_estimators, test_batch_count=test_batch_count, input_size=input_size, compact_factor=dim_compact_factor, depth=depth, tree_method=tree_method)











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