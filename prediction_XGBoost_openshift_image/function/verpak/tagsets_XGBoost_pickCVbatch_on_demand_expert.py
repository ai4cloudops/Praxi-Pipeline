import os, pickle, time, gc
import yaml
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import sklearn.metrics as metrics
import scipy
from pathlib import Path
import multiprocessing as mp
import xgboost as xgb
import sys
from sklearn.feature_extraction.text import CountVectorizer

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
            # if not inference_flag:
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

def has_intersection(from_set, to_set):
    for ele in from_set:
        if ele in to_set:
            return True
    return False

def get_intersection(from_set, to_set):
    ret = []
    for ele in from_set:
        if ele in to_set:
            ret.append(ele)
    return ret

def tagsets_to_matrix(tags_path, tag_files_l = None, index_tag_mapping_path=None, tag_index_mapping_path=None, index_label_mapping_path=None, label_index_mapping_path=None, cwd="/home/cc/Praxi-study/Praxi-Pipeline/prediction_base_image/model_testing_scripts/cwd/", train_flag=False, inference_flag=True, iter_flag=False, packages_select_set=set(), tokens_filter_set=set(), input_size=None, compact_factor=1, freq=100, all_tags_set=None,all_label_set=None,tags_by_instance_l=None,labels_by_instance_l=None,tagset_files=None, feature_importance=np.array([])):
    op_durations = defaultdict(int)
    if index_tag_mapping_path == None:
        index_tag_mapping_path=cwd+'index_tag_mapping'
        tag_index_mapping_path=cwd+'tag_index_mapping'
        index_label_mapping_path=cwd+'index_label_mapping'
        label_index_mapping_path=cwd+'label_index_mapping'

        index_tag_mapping_iter_path=cwd+"index_tag_mapping_iter"
        tag_index_mapping_iter_path=cwd+"tag_index_mapping_iter"
        index_label_mapping_iter_path=cwd+"index_label_mapping_iter"
        label_index_mapping_iter_path=cwd+"label_index_mapping_iter"
    
    # if all_tags_set == None:
    #     all_tags_set, all_label_set = set(), set()
    #     tags_by_instance_l, labels_by_instance_l = [], []
    #     tagset_files = []

    #     # # debuging with lcoal tagset files
    #     # for tag_file in tqdm(os.listdir(tags_path)):
    #     #     data_instance_d = read_tokens(tags_path, tag_file, cwd, packages_select_set, inference_flag)
    #     #     if len(data_instance_d) ==4:
    #     #         tagset_files.append(data_instance_d['tag_file'])
    #     #         all_tags_set.update(data_instance_d['local_all_tags_set'])
    #     #         tags_by_instance_l.append(data_instance_d['instance_feature_tags_d'])
    #     #         all_label_set.update(data_instance_d['labels'])
    #     #         labels_by_instance_l.append(data_instance_d['labels'])


    #     if tag_files_l == None:
    #         tag_files_l = [tag_file for tag_file in os.listdir(tags_path) if (tag_file[-3:] == 'tag') and (tag_file[:-4].rsplit('-', 1)[0] in packages_select_set or packages_select_set == set())]
    #     # return 
    #     tag_files_l_of_l, step = [], len(tag_files_l)//mp.cpu_count()+1
    #     for i in range(0, len(tag_files_l), step):
    #         tag_files_l_of_l.append(tag_files_l[i:i+step])
    #     # pool = mp.Pool(processes=mp.cpu_count())
    #     pool = mp.Pool(processes=2)
    #     data_instance_d_l = [pool.apply_async(map_tagfilesl, args=(tags_path, tag_files_l, cwd, inference_flag, freq), kwds={"tokens_filter_set": tokens_filter_set}) for tag_files_l in tqdm(tag_files_l_of_l)]
    #     data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
    #     pool.close()
    #     pool.join()
    #     for data_instance_d in data_instance_d_l:
    #         if len(data_instance_d) == 5:
    #                 tagset_files.extend(data_instance_d['tagset_files'])
    #                 all_tags_set.update(data_instance_d['all_tags_set'])
    #                 tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
    #                 all_label_set.update(data_instance_d['all_label_set'])
    #                 labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
    #     # for data_instance_d in data_instance_d_l:
    #     #     if len(data_instance_d) == 5:
    #     #             tagset_files.extend(data_instance_d['tagset_files'])
    #     #             all_tags_set.update(data_instance_d['all_tags_set'])
    #     #             tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
    #     #             all_label_set.update(data_instance_d['all_label_set'])
    #     #             labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
    #     # for data_instance_d in data_instance_d_l:
    #     #     if len(data_instance_d) == 5:
    #     #             tagset_files.extend(data_instance_d['tagset_files'])
    #     #             all_tags_set.update(data_instance_d['all_tags_set'])
    #     #             tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
    #     #             all_label_set.update(data_instance_d['all_label_set'])
    #     #             labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])

    #     # pool = mp.Pool(processes=mp.cpu_count())
    #     # # pool = mp.Pool(processes=1)
    #     # data_instance_d_l = [pool.apply_async(read_tokens, args=(tags_path, tag_file, cwd, packages_select_set, inference_flag)) for tag_file in tqdm(os.listdir(tags_path))]
    #     # data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
    #     # for data_instance_d in data_instance_d_l:
    #     #     if len(data_instance_d) ==4:
    #     #             tagset_files.append(data_instance_d['tag_file'])
    #     #             all_tags_set.update(data_instance_d['local_all_tags_set'])
    #     #             tags_by_instance_l.append(data_instance_d['instance_feature_tags_d'])
    #     #             all_label_set.update(data_instance_d['labels'])
    #     #             labels_by_instance_l.append(data_instance_d['labels'])

    #     # # kfp / RHODS pipeline with intermediate data as dumps 
    #     # with open(tags_path, 'rb') as reader:
    #     #     tagsets_l = pickle.load(reader)
    #     #     for tagset in tagsets_l:
    #     #         instance_feature_tags_d = defaultdict(int)
    #     #         # feature 
    #     #         for tag_vs_count in tagset['tags']:
    #     #             k,v = tag_vs_count.split(":")
    #     #             all_tags_set.add(k)
    #     #             instance_feature_tags_d[k] += int(v)
    #     #         tags_by_instance_l.append(instance_feature_tags_d)
    #     #         # label
    #     #         if not inference_flag:
    #     #             if 'labels' in tagset:
    #     #                 all_label_set.update(tagset['labels'])
    #     #                 labels_by_instance_l.append(tagset['labels'])
    #     #             else:
    #     #                 all_label_set.add(tagset['label'])
    #     #                 labels_by_instance_l.append([tagset['label']])
            
    #     # with open(cwd+'tagset_files.yaml', 'w') as f:
    #     #     yaml.dump(tagset_files, f)
    #     #     # for line in tagset_files:
    #     #     #     f.write(f"{line}\n")

    #     # Sorting instances
    #     if not inference_flag:
    #         zipped = list(zip(tagset_files, tags_by_instance_l, labels_by_instance_l))
    #         zipped.sort(key=lambda x: x[0])
    #         tagset_files, tags_by_instance_l, labels_by_instance_l = zip(*zipped)
    #     else:
    #         zipped = list(zip(tagset_files, tags_by_instance_l))
    #         zipped.sort(key=lambda x: x[0])
    #         tagset_files, tags_by_instance_l = zip(*zipped)



    # # #############
    #     # # Save tag:count in mapping format
    #     # with open(tags_path+"all_tags_set.obj","wb") as filehandler:
    #     #      pickle.dump(all_tags_set, filehandler)
    #     # with open(tags_path+"all_label_set.obj","wb") as filehandler:
    #     #      pickle.dump(all_label_set, filehandler)
    #     # with open(tags_path+"tags_by_instance_l.obj","wb") as filehandler:
    #     #      pickle.dump(tags_by_instance_l, filehandler)
    #     # with open(tags_path+"labels_by_instance_l.obj","wb") as filehandler:
    #     #      pickle.dump(labels_by_instance_l, filehandler)
    #     # with open(tags_path+"tagset_files.obj","wb") as filehandler:
    #     #      pickle.dump(tagset_files, filehandler)

    #     # # Load tag:count in mapping format 
    #     # with open(tags_path+"all_tags_set.obj","rb") as filehandler:
    #     #     all_tags_set = pickle.load(filehandler)
    #     # with open(tags_path+"all_label_set.obj","rb") as filehandler:
    #     #     all_label_set = pickle.load(filehandler)
    #     # with open(tags_path+"tags_by_instance_l.obj","rb") as filehandler:
    #     #     tags_by_instance_l = pickle.load(filehandler)
    #     # with open(tags_path+"labels_by_instance_l.obj","rb") as filehandler:
    #     #     labels_by_instance_l = pickle.load(filehandler)
    #     # with open(tags_path+"tagset_files.obj","rb") as filehandler:
    #     #     tagset_files = pickle.load(filehandler)
    # # #############

    # Feature Matrix Generation
    t_gen_mapping_0 = time.time()
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
    t_gen_mapping_t = time.time()
    op_durations["gen_mapping"] = t_gen_mapping_t-t_gen_mapping_0
    op_durations["len(all_tags_l)"] = len(all_tags_l)
    ## Generate Feature Matrix
    t_gen_mat_0 = time.time()
    t_get_feature_0 = time.time()
    instance_row_list, instance_row_idx_set  = [], []
    # instance_row_idx_set, used_row_count  = [], 0
    values_l, pos_x_l, pos_y_l = [],[],[]
    used_tags_set = set([all_tags_l[used_fidx] for used_fidx in np.where(feature_importance > 0)[0].tolist()])
    t_get_feature_t = time.time()
    op_durations["get_feature"] = t_get_feature_t-t_get_feature_0
    for instance_row_idx, instance_tags_d in enumerate(tags_by_instance_l):
        if input_size == None:
            input_size = len(all_tags_l)//compact_factor
        t_selector_0 = time.time()
        # token_intersection = set(instance_tags_d.keys()).intersection(used_tags_set)
        # if (feature_importance.size != 0) and not token_intersection:
        # if (feature_importance.size != 0) and used_tags_set.isdisjoint(instance_tags_d.keys()):
        # if (feature_importance.size != 0) and not has_intersection(used_tags_set, instance_tags_d.keys()):
        used_instance_tags_list = get_intersection(used_tags_set, instance_tags_d.keys())
        if (feature_importance.size != 0) and not used_instance_tags_list:
            t_selector_t = time.time()
            op_durations["selector"] += t_selector_t-t_selector_0
            continue
        t_selector_t = time.time()
        op_durations["selector"] += t_selector_t-t_selector_0
        t_mat_builder_0 = time.time()
        instance_row = np.zeros(input_size)
        # for tag_name,tag_count in instance_tags_d.items():
        #     if tag_name in tag_index_mapping:  # remove new tags unseen in mapping.
        #         instance_row[tag_index_mapping[tag_name]%input_size] = tag_count
        #         # values_l.append(tag_count)
        #         # pos_x_l.append(tag_index_mapping[tag_name]%input_size)
        #         # pos_y_l.append(used_row_count)
        #     # else:
        #     #     removed_tags_l.append(tag_name)
        # for tag_name,tag_col_idx in tag_index_mapping.items():
        #     if tag_name in instance_tags_d:  # remove new tags unseen in mapping.
        #         instance_row[tag_col_idx%input_size] = instance_tags_d[tag_name]
        #         # values_l.append(tag_count)
        #         # pos_x_l.append(tag_index_mapping[tag_name]%input_size)
        #         # pos_y_l.append(used_row_count)
        #     # else:
        #     #     removed_tags_l.append(tag_name)

        #  !!!!!!!!!!!!!!!!!!!!!!! 
        for tag_name in used_instance_tags_list:
            instance_row[tag_index_mapping[tag_name]%input_size] = instance_tags_d[tag_name]
            # values_l.append(tag_count)
            # pos_x_l.append(tag_index_mapping[tag_name]%input_size)
            # pos_y_l.append(used_row_count)
        else:
            # used_row_count += 1
            instance_row_idx_set.append(instance_row_idx)
            instance_row_list.append(scipy.sparse.csr_matrix(instance_row))
        t_mat_builder_t = time.time()
        op_durations["mat_builder"] += t_mat_builder_t-t_mat_builder_0
    instance_row_count = instance_row_idx+1
    instance_row_idx_set = set(instance_row_idx_set)

    t_list_to_mat_0 = time.time()
    if instance_row_list:
        feature_matrix = scipy.sparse.vstack(instance_row_list)
    else:
        feature_matrix = scipy.sparse.csr_matrix([])
    # if values_l:
    #     feature_matrix = scipy.sparse.coo_matrix((values_l, (pos_y_l, pos_x_l)), shape=(used_row_count, input_size)).tocsr()
    # else:
    #     feature_matrix = scipy.sparse.csr_matrix([])
    # # feature_matrix = np.vstack(instance_row_list)
    # del instance_row_list
    t_list_to_mat_t = time.time()
    t_gen_mat_t = time.time()
    op_durations["list_to_mat"] = t_list_to_mat_t-t_list_to_mat_0
    op_durations["gen_mat"] = t_gen_mat_t-t_gen_mat_0
    # with open(cwd+'removed_tags_l', 'wb') as fp:
    #     pickle.dump(removed_tags_l, fp)
    # with open(cwd+'removed_tags_l.txt', 'w') as f:
    #     for line in removed_tags_l:
    #         f.write(f"{line}\n")
    


    # Label Matrix Generation
    label_matrix = np.array([])
    # if not inference_flag:
    #     removed_label_l = []
    #     ## Handling Mapping

    #     if train_flag and not iter_flag:  # generate initial mapping.
    #         all_label_l = list(all_label_set)
    #         label_index_mapping = {}
    #         for idx, tag in enumerate(all_label_l):
    #             label_index_mapping[tag] = idx
    #         with open(index_label_mapping_path, 'wb') as fp:
    #             pickle.dump(all_label_l, fp)
    #         with open(label_index_mapping_path, 'wb') as fp:
    #             pickle.dump(label_index_mapping, fp)
    #         # with open(cwd+'removed_tags_l.txt', 'w') as f:
    #         #     for line in all_label_l:
    #         #         f.write(f"{line}\n")
    #     elif train_flag and iter_flag:  # adding mapping.
    #         with open(index_label_mapping_path, 'rb') as fp:
    #             loaded_all_label_l = pickle.load(fp)
    #             loaded_all_label_set = set(loaded_all_label_l)
    #             new_label_set = all_label_set.difference(loaded_all_label_set)
    #             all_label_l = loaded_all_label_l + list(new_label_set)
    #             with open(index_label_mapping_iter_path, 'wb') as fp:
    #                 pickle.dump(all_label_l, fp)
    #         with open(label_index_mapping_path, 'rb') as fp:
    #             label_index_mapping = pickle.load(fp)
    #             for idx, tag in enumerate(all_label_l[len(loaded_all_label_l):]):
    #                 label_index_mapping[tag] = idx+len(loaded_all_label_l)
    #             with open(label_index_mapping_iter_path, 'wb') as fp:
    #                 pickle.dump(label_index_mapping, fp)
    #     elif not train_flag and iter_flag:  # load iter mapping.
    #         with open(index_label_mapping_iter_path, 'rb') as fp:
    #             all_label_l = pickle.load(fp)
    #         with open(label_index_mapping_iter_path, 'rb') as fp:
    #             label_index_mapping = pickle.load(fp)
    #         with open(cwd+'loaded_index_label_mapping_iter.txt', 'w') as f:
    #             for line in all_label_l:
    #                 f.write(f"{line}\n")
    #     else:  # not train_flag and not iter_flag: load initial mapping.
    #         with open(index_label_mapping_path, 'rb') as fp:
    #             all_label_l = pickle.load(fp)
    #         with open(label_index_mapping_path, 'rb') as fp:
    #             label_index_mapping = pickle.load(fp)
    #         with open(cwd+'loaded_index_label_mapping.txt', 'w') as f:
    #             for line in all_label_l:
    #                 f.write(f"{line}\n")
    #     ## Handling Label Matrix
    #     instance_row_list = []
    #     # label_matrix = np.zeros(len(all_label_l))
    #     for instance_row_idx, labels in enumerate(labels_by_instance_l):
    #         if instance_row_idx not in instance_row_idx_set:
    #             continue
    #         instance_row = np.zeros(len(all_label_l))
    #         for label in labels:
    #             if label in label_index_mapping:    # remove new labels
    #                 instance_row[label_index_mapping[label]] = 1
    #             else:
    #                 removed_label_l.append(label)
    #         else:
    #             # instance_row_list.append(scipy.sparse.csr_matrix(instance_row))
    #             instance_row_list.append(instance_row)
    #             # label_matrix = np.vstack([label_matrix, instance_row])
    #     if instance_row_list:
    #         label_matrix = np.vstack(instance_row_list)
    #     else:
    #         label_matrix = np.array([])
    #     with open(cwd+'removed_label_l', 'wb') as fp:
    #         pickle.dump(removed_label_l, fp)
    #     with open(cwd+'removed_label_l.txt', 'w') as f:
    #         for line in removed_label_l:
    #             f.write(f"{line}\n")
    
    return [tagset_files[instance_row_idx] for instance_row_idx in list(instance_row_idx_set)], feature_matrix, label_matrix, instance_row_idx_set, instance_row_count, op_durations
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




def run_pred(cwd, clf_path_l, test_tags_path, n_jobs=64, n_estimators=100, packages_select_set=set(), test_batch_count=1, input_size=None, compact_factor=1, depth=1, tree_method="auto"):
    # # cwd = "/pipelines/component/cwd/"
    # cwd = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd/"
    # clf_path = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd/model_init.json"
    # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/inference_test/"
    Path(cwd).mkdir(parents=True, exist_ok=True)


    all_tags_set,all_label_set,tags_by_instance_l,labels_by_instance_l,tagset_files = None,None,None,None,None
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
        print(cwd)
        op_durations[clf_path+"\n BOW_XGB.predictclf"+str(clf_idx)] = 0 
        with open(clf_path[:-15]+'index_label_mapping', 'rb') as fp:
            clf_labels_l = pickle.load(fp)
            labels_list.append(np.array(clf_labels_l))
        t0 = time.time()
        BOW_XGB = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=depth, learning_rate=0.1,silent=False, objective='binary:logistic', \
                        booster='gbtree', n_jobs=n_jobs, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                        subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, tree_method=tree_method)
        BOW_XGB.load_model(clf_path)
        BOW_XGB.set_params(n_jobs=n_jobs)
        feature_importance = BOW_XGB.feature_importances_
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
            tagset_files_used, feature_matrix, label_matrix, instance_row_idx_set, instance_row_count = tagsets_to_matrix(test_tags_path, tag_files_l = tag_files_l[batch_first_idx:batch_first_idx+step], inference_flag=False, cwd=clf_path[:-15], packages_select_set=packages_select_set, input_size=input_size, compact_factor=compact_factor, all_tags_set=all_tags_set,all_label_set=all_label_set,tags_by_instance_l=tags_by_instance_l,labels_by_instance_l=labels_by_instance_l,tagset_files=tagset_files, feature_importance=feature_importance) # get rid of "model_init.json" in the clf_path.
            # # ########### load a previously converted encoding format data obj
            # with open(test_tags_path+"feature_matrix.obj","rb") as filehandler:
            #     feature_matrix = pickle.load(filehandler)
            # with open(test_tags_path+"label_matrix.obj","rb") as filehandler:
            #     label_matrix = pickle.load(filehandler)
            # with open(test_tags_path+"tagset_files_used.obj","rb") as filehandler:
            #     tagset_files_used = pickle.load(filehandler)
            # # ############################################
            t1 = time.time()
            if feature_matrix.size != 0:
                op_durations[clf_path+"\n tagsets_to_matrix-testset"+str(batch_first_idx)+"/"+str(test_batch_count)] = t1-t0
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
                op_durations[clf_path+"\n BOW_XGB.predictclf"+str(clf_idx)] += t1-t0
                op_durations[clf_path+"\n one_hot_to_names_"+str(batch_first_idx)+"/"+str(test_batch_count)] = t2-t1
            else:
                op_durations[clf_path+"\n tagsets_to_matrix-testset"+str(batch_first_idx)+"/"+str(test_batch_count)] = 0
                op_durations[clf_path+"\n feature_matrix_size_0_"+str(batch_first_idx)+"/"+str(test_batch_count)] = 0
                op_durations[clf_path+"\n feature_matrix_size_1_"+str(batch_first_idx)+"/"+str(test_batch_count)] = 0
                op_durations[clf_path+"\n label_matrix_size_0_"+str(batch_first_idx)+"/"+str(test_batch_count)] = 0
                op_durations[clf_path+"\n label_matrix_size_1_"+str(batch_first_idx)+"/"+str(test_batch_count)] = 0

                op_durations[clf_path+"\n BOW_XGB.predict_"+str(batch_first_idx)+"/"+str(test_batch_count)] = 0
                op_durations[clf_path+"\n BOW_XGB.predictclf"+str(clf_idx)] += 0
                op_durations[clf_path+"\n one_hot_to_names_"+str(batch_first_idx)+"/"+str(test_batch_count)] = 0


            # !!!!!!!!!!!!!!!!!!!!!!!! fill zeros for samples without recogonizable features by this clf
            all_label_len = len(clf_labels_l)
            new_label_matrix = np.vstack(np.zeros((instance_row_count, all_label_len)))
            for instance_row_idx, new_instance_row_idx in enumerate(list(instance_row_idx_set)):
                new_label_matrix[new_instance_row_idx, :] = label_matrix[instance_row_idx, :]
            label_matrix = new_label_matrix

            new_pred_label_matrix = np.vstack(np.zeros((instance_row_count, all_label_len)))
            for instance_row_idx, new_instance_row_idx in enumerate(list(instance_row_idx_set)):
                new_pred_label_matrix[new_instance_row_idx, :] = pred_label_matrix[instance_row_idx, :]
            pred_label_matrix = new_pred_label_matrix

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

# if __name__ == "__main__":
#     ##################################
#     n_samples_d = {}
#     test_portion_d = {}
#     # ============= data_0
#     n_samples_d["data_0"]=25
#     test_portion_d["data_0"]=0.2

#     # ============= data_3
#     n_samples_d["data_3"]=21
#     test_portion_d["data_3"]=0.2

#     # ============= data_4
#     n_samples_d["data_4"]=4
#     test_portion_d["data_4"]=0.25
#     # ###################### choose CV batch ######################
#     for test_sample_batch_idx in [0]:   # this is the initial idx of a test batch, i.e., if test batch size is 4*0.25 = 1, `test_sample_batch_idx`` means pick the 0-index element as the test batch.
#         # ###################################
#         # run_pred()
#         # Testing the ML dataset
#         for dataset in ["data_4"]:
#             n_samples = n_samples_d[dataset]
#             for (with_filter, freq) in [[(False, 100)][0]]:
#                 for n_jobs in [32]:
#                     for clf_njobs in [32]:
#                         for n_models, test_batch_count in zip([[1000, 750, 500][2]],[1,1,1,1]): # zip([50,25,20,15,10,5,1],[1,1,1,1,1,1,8]):
#                             for n_estimators in [100]:
#                                 for depth in [1]:
#                                     for tree_method in["exact"]: # "exact","approx","hist"
#                                         for max_bin in [1]:
#                                             for input_size, dim_compact_factor in zip([None],[1,1,1,1,1]):
#                                                 for shuffle_idx in range(3):

#                                                     clf_path = []
#                                                     for i in range(n_models):
#                                                         clf_pathname = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(clf_njobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/model_init.json"
#                                                         if os.path.isfile(clf_pathname):
#                                                             clf_path.append(clf_pathname)
#                                                         else:
#                                                             print("clf is missing!")
#                                                             sys.exit(-1)
#                                                     cwd = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak_on_demand_expert/"
#                                                     # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/big_ML_biased_test/"
#                                                     test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/"+dataset+"/tagsets_ML/"
#                                     #    run_init_train(train_tags_path, test_tags_path, cwd, n_jobs=n_jobs, n_estimators=n_estimators, train_packages_select_set=train_subset, test_packages_select_set=test_subset, input_size=input_size, depth=depth, tree_method=tree_method)
#                                                     run_pred(cwd, clf_path, test_tags_path, n_jobs=n_jobs, n_estimators=n_estimators, test_batch_count=test_batch_count, input_size=input_size, compact_factor=dim_compact_factor, depth=depth, tree_method=tree_method)








if __name__ == "__main__":
    op_durations = defaultdict(int)
    t_0 = time.time()
    # test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML_test_120/"
    test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML/"

    # cwd = "/home/cc/test"

    dataset = "data_4"
    n_models = 1
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
    cwd = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts_online/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak_on_demand_expert/"
    cwd_clf = "/home/cc/test"
    Path(cwd).mkdir(parents=True, exist_ok=True)


    # Data 
    t_data_0 = time.time()
    tag_files_l = [tag_file for tag_file in os.listdir(test_tags_path) if tag_file[-3:] == 'tag']
    tag_files_l_of_l, step = [], len(tag_files_l)//mp.cpu_count()+1
    for i in range(0, len(tag_files_l), step):
        tag_files_l_of_l.append(tag_files_l[i:i+step])
    pool = mp.Pool(processes=32)
    data_instance_d_l = [pool.apply_async(map_tagfilesl, args=(test_tags_path, tag_files_l, cwd, True, freq)) for tag_files_l in tqdm(tag_files_l_of_l)]
    data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
    pool.close()
    pool.join()
    all_tags_set, all_label_set = set(), set()
    tags_by_instance_l, labels_by_instance_l = [], []
    tagset_files = []
    for data_instance_d in data_instance_d_l:
        if len(data_instance_d) == 5:
                tagset_files.extend(data_instance_d['tagset_files'])
                all_tags_set.update(data_instance_d['all_tags_set'])
                tags_by_instance_l.extend(data_instance_d['tags_by_instance_l'])
                all_label_set.update(data_instance_d['all_label_set'])
                labels_by_instance_l.extend(data_instance_d['labels_by_instance_l'])
    t_data_t = time.time()
    op_durations["total_data_load__time"] = t_data_t-t_data_0

    # Models
    t_clf_path_0 = time.time()
    clf_path_l = []
    for i in range(n_models):
        clf_pathname = f"{cwd_clf}/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(clf_njobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/model_init.json"
        if os.path.isfile(clf_pathname):
            clf_path_l.append(clf_pathname)
        else:
            print(f"clf is missing: {clf_pathname}")
            sys.exit(-1)
    t_clf_path_t = time.time()
    op_durations["total_clf_path_load__time"] = t_clf_path_t-t_clf_path_0

    # Make inference
    # label_matrix_list, pred_label_matrix_list, labels_list = [], [], []
    # values_l_, pos_x_l_, pos_y_l_ = [],[],[]
    results = defaultdict(list)
    for clf_idx, clf_path in enumerate(clf_path_l):
        with open(clf_path[:-15]+'index_label_mapping', 'rb') as fp:
            clf_labels_l = pickle.load(fp)
            # labels_list.append(np.array(clf_labels_l))


        t_per_clf_0 = time.time()

        BOW_XGB = xgb.XGBClassifier(max_depth=10, learning_rate=0.1,silent=False, objective='binary:logistic', \
                        booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                        subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1)
        BOW_XGB.load_model(clf_path)
        BOW_XGB.set_params(n_jobs=n_jobs)
        feature_importance = BOW_XGB.feature_importances_

        t_per_clf_loading_t = time.time()
        op_durations[f"clf{clf_idx}_load_time"] = t_per_clf_loading_t-t_per_clf_0
        op_durations["total_clf_load__time"] += t_per_clf_loading_t-t_per_clf_0

        # label_matrix_list_per_clf, pred_label_matrix_list_per_clf = [],[]
        pred_label_matrix_list_per_clf = []
        step = len(tag_files_l)
        for batch_first_idx in range(0, len(tag_files_l), step):
            t_encoder_0 = time.time()
            tagset_files_used, feature_matrix, label_matrix, instance_row_idx_set, instance_row_count, encoder_op_durations = tagsets_to_matrix(test_tags_path, tag_files_l = tag_files_l, cwd=clf_path[:-15], all_tags_set=all_tags_set,all_label_set=all_label_set,tags_by_instance_l=tags_by_instance_l,labels_by_instance_l=labels_by_instance_l,tagset_files=tagset_files, feature_importance=feature_importance)
            # values_l_.extend(values_l)
            # pos_x_l_.extend(pos_x_l)
            # pos_y_l_.extend(pos_y_l)
            t_encoder_t = time.time()
            op_durations[f"encoder{clf_idx}_op_durations"] = encoder_op_durations
            op_durations[f"encoder{clf_idx}_time"] += t_encoder_t-t_encoder_0
            op_durations["total_encoder_time"] += t_encoder_t-t_encoder_0
            t_inference_0 = time.time()
            if feature_matrix.size != 0:
                # prediction
                pred_label_matrix = BOW_XGB.predict(feature_matrix)
            t_inference_t = time.time()
            op_durations[f"inference{clf_idx}_time"] += t_inference_t-t_inference_0
            op_durations["total_inference_time"] += t_inference_t-t_inference_0
            # !!!!!!!!!!!!!!!!!!!!!!!! fill zeros for samples without recogonizable features by this clf
            all_label_len = len(clf_labels_l)
            # new_label_matrix = np.vstack(np.zeros((instance_row_count, all_label_len)))
            # for instance_row_idx, new_instance_row_idx in enumerate(list(instance_row_idx_set)):
            #     new_label_matrix[new_instance_row_idx, :] = label_matrix[instance_row_idx, :]
            # label_matrix = new_label_matrix

            new_pred_label_matrix = np.vstack(np.zeros((instance_row_count, all_label_len)))
            for instance_row_idx, new_instance_row_idx in enumerate(list(instance_row_idx_set)):
                new_pred_label_matrix[new_instance_row_idx, :] = pred_label_matrix[instance_row_idx, :]
            pred_label_matrix = new_pred_label_matrix

            # label_matrix_list_per_clf.append(label_matrix)
            pred_label_matrix_list_per_clf.append(pred_label_matrix)

        
        # label_matrix_list_per_clf = np.vstack(label_matrix_list_per_clf)
        pred_label_matrix_list_per_clf = np.vstack(pred_label_matrix_list_per_clf)
        results = merge_preds(results, one_hot_to_names(clf_path[:-15]+'index_label_mapping', pred_label_matrix_list_per_clf))
        # label_matrix_list.append(label_matrix_list_per_clf)
        # pred_label_matrix_list.append(pred_label_matrix_list_per_clf)


        
        t_per_clf_t = time.time()
        op_durations[f"clf{clf_idx}_time"] = t_per_clf_t-t_per_clf_0
        op_durations["total_clf_time"] += t_per_clf_t-t_per_clf_0
        print("clf"+str(clf_idx)+" pred done")
    
    
    t_t = time.time()
    # op_durations["len(values_l_)"] = len(values_l_)
    # op_durations["len(pos_x_l_)"] = len(pos_x_l_)
    # op_durations["len(pos_y_l_)"] = len(pos_y_l_)
    op_durations["total_time"] = t_t-t_0
    with open(cwd+"metrics.yaml", 'w') as writer:
        yaml.dump(op_durations, writer)
    print(results)
    # label_matrix = np.hstack(label_matrix_list)
    # pred_label_matrix = np.hstack(pred_label_matrix_list)
    # labels = np.hstack(labels_list)
    # print_metrics(cwd, 'metrics_pred.out', label_matrix, pred_label_matrix, labels, op_durations)
    # print(one_hot_to_names(f"{clf_path[:-15]}index_label_mapping", pred_label_matrix))






# if __name__ == "__main__":
#     t0 = time.time()
#     from sklearn.feature_extraction.text import CountVectorizer

#     test_tags_path = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_ML_test/"

#     cwd = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts"

#     dataset = "data_4"
#     n_models = 1
#     shuffle_idx = 0
#     test_sample_batch_idx = 0
#     n_samples = 4
#     clf_njobs = 32
#     n_estimators = 100
#     depth = 1
#     input_size = None
#     dim_compact_factor = 1
#     tree_method = "exact"
#     max_bin = 1
#     with_filter = True
#     freq = 25


#     # Data 
#     tag_files_l = [tag_file for tag_file in os.listdir(test_tags_path) if tag_file[-3:] == 'tag']
#     tag_files_l_of_l, step = [], len(tag_files_l)//mp.cpu_count()+1
#     for i in range(0, len(tag_files_l), step):
#         tag_files_l_of_l.append(tag_files_l[i:i+step])
#     pool = mp.Pool(processes=2)
#     data_instance_d_l = [pool.apply_async(map_tagfilesl, args=(test_tags_path, tag_files_l, cwd, True, freq)) for tag_files_l in tqdm(tag_files_l_of_l)]
#     data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
#     pool.close()
#     pool.join()
#     tags_by_instance_l = []
#     for data_instance_d in data_instance_d_l:
#         if len(data_instance_d) == 5:
#             tags_by_instance_l.extend(data_instance_d['tags_by_instance_l']) 
#     observations_prepared = []
#     for observation in tags_by_instance_l:
#         observation_str = ' '.join([' '.join([token] * count) for token, count in observation.items()])
#         observations_prepared.append(observation_str)


#     # Clf
#     clf_path_l = []
#     for i in range(n_models):
#         # clf_pathname = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(clf_njobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/model_init.json"
#         #                '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_  data4    _      1000       _      0   _train_      0             shuffleidx_      0                       testsamplebatchidx_      4           nsamples_      32          njobs_      100            trees_      1       depth_      None         -      1                    rawinput_sampling1_      exact         treemethod_      1         maxbin_modize_par_      True            25   removesharedornoisestags_verpak/model_init.json'
#         clf_pathname = f"{cwd}/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(clf_njobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/model_init.json"
#         if os.path.isfile(clf_pathname):
#             clf_path_l.append(clf_pathname)
#         else:
#             print(f"clf is missing: {clf_pathname}")
#             sys.exit(-1)


#     # Make Prediction
#     results = defaultdict(list)
#     for clf_idx, clf_path in enumerate(clf_path_l):
#         BOW_XGB = xgb.XGBClassifier(max_depth=10, learning_rate=0.1,silent=False, objective='binary:logistic', \
#                         booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
#                         subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1)
#         BOW_XGB.load_model(clf_path)
#         feature_importance = BOW_XGB.feature_importances_

#         with open(clf_path[:-15]+'tag_index_mapping', 'rb') as fp:
#             tag_index_mapping = pickle.load(fp)
#         vectorizer = CountVectorizer(vocabulary=tag_index_mapping)

#         feature_matrix_scikit = vectorizer.transform(observations_prepared)
#         # if feature_matrix.size != 0:
#         #     # prediction
#         #     pred_label_matrix = BOW_XGB.predict(feature_matrix)
#         #     results = merge_preds(results, one_hot_to_names(f"{clf_path[:-15]}index_label_mapping", pred_label_matrix))
        


#         tag_files_l = [tag_file for tag_file in os.listdir(test_tags_path) if tag_file[-3:] == 'tag']
#         step = len(tag_files_l)
#         for batch_first_idx in range(0, len(tag_files_l), step):
#             tagset_files_used, feature_matrix, label_matrix, instance_row_idx_set, instance_row_count = tagsets_to_matrix(test_tags_path, tag_files_l = tag_files_l[batch_first_idx:batch_first_idx+step], cwd=clf_path[:-15], feature_importance=feature_importance)
#             # tagset_files, feature_matrix, label_matrix = tagsets_XGBoost.tagsets_to_matrix(test_tags_path, index_tag_mapping_path, tag_index_mapping_path, index_label_mapping_path, label_index_mapping_path, train_flag=False, cwd=cwd)


        
#         print("clf"+str(clf_idx)+" pred done")
    
#     t1 = time.time()
#     print(f"total time: {t1-t0}")
#     print(results)







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