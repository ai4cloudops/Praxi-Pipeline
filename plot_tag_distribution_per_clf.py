import os
import sys
import time
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import yaml, json, pickle, statistics
from tqdm import tqdm
from collections import defaultdict, Counter
from pathlib import Path
import multiprocessing as mp
import itertools
import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42	
# matplotlib.rcParams['text.usetex'] = True

def read_tagset(tagset_pathname):
    with open(tagset_pathname, "r") as stream:
        try:
            tagset = yaml.safe_load(stream)
            if 'labels' in tagset:
                label_count = str(len(tagset['labels']))
                labels_str = '&&'.join(tagset['labels'])
                tags_d, tagnames_set = {}, set()
            else:
                label_count = "1"
                labels_str = tagset['label']
            # tags_d, tagnames_set = {}, set()
            # for tag in tagset['tags']:
            #     tag_l = tag.split(":")
            #     tags_d[tag_l[0]] = tag_l[1]
            #     tagnames_set.add(tag_l[0])
            tagnames_set = set(tagset['tags'].keys())
            tags_d = tagset['tags']
            tags_length = len(tagset['tags'])
            return {"label_count":label_count, "labels_str": labels_str, "tags_length": tags_length, "tags_d": tags_d, "tagnames_set": tagnames_set}
        except yaml.YAMLError as exc:
            print(exc)

def plot_size():
    target_dir = "tagsets_SL"
    dirname = "/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/"
    out_dirname = dirname+target_dir+"/"
    # print(out_dirname)
    tagsetfilenames_l = [name for name in os.listdir(out_dirname) if os.path.isfile(out_dirname+name) and name[-4:]!=".obj"]
    # print(tagsetfilenames_l)
    # if len(tagsetfilenames_l) == 2:
    pool = mp.Pool(processes=mp.cpu_count())
    data_instance_d_l = [pool.apply_async(read_tagset, args=(out_dirname+tagsets_name,)) for tagsets_name in tqdm(tagsetfilenames_l)]
    data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
    pool.close()
    pool.join()

    clf_path = []
    for test_sample_batch_idx in [0]:   # this is the initial idx of a test batch, i.e., if test batch size is 4*0.25 = 1, `test_sample_batch_idx`` means pick the 0-index element as the test batch.
        for dataset in ["data_4"]:
            n_samples = 4
            for (with_filter, freq) in [(True, 300), (True, 100)]:
                for n_jobs in [32]:
                    for clf_njobs in [32]:
                        for n_models, test_batch_count in zip([50, 25, 10, 1],[1,1,1,1]): # zip([50,25,20,15,10,5,1],[1,1,1,1,1,1,8]):
                            for n_estimators in [100]:
                                for depth in [1]:
                                    for tree_method in["exact"]: # "exact","approx","hist"
                                        for max_bin in [1]:
                                            for input_size, dim_compact_factor in zip([None],[1,1,1,1,1]):
                                                for shuffle_idx in range(1,3):
                                                    for i in range(n_models):
                                                        clf_pathname = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(clf_njobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/model_init.json"
                                                        if os.path.isfile(clf_pathname):
                                                            clf_path.append(clf_pathname)
                                                        else:
                                                            print("clf is missing!")
                                                            sys.exit(-1)
    for clf_idx, clf_path in enumerate(clf_path):
        
        index_tag_mapping_path=clf_path[:-15]+'index_tag_mapping'
        index_label_mapping_path=clf_path[:-15]+'index_label_mapping'

        with open(index_tag_mapping_path, 'rb') as fp:
            all_tags_set = set(pickle.load(fp))
        with open(index_label_mapping_path, 'rb') as fp:
            all_label_set = set(pickle.load(fp))

        label_tagsnameset_d = defaultdict(set)
        for data_instance_d in data_instance_d_l:
            if data_instance_d["labels_str"] in all_label_set:
                label_tagsnameset_d[data_instance_d["labels_str"]].update(set(data_instance_d["tags_d"].keys())) # {"label":{"tagname",},}

        dirname = clf_path
        plot_dir = "plots/"
        Path(dirname+plot_dir).mkdir(parents=True, exist_ok=True)


        tagnames_set = all_tags_set
        tagnames_reoccurentcount_d = defaultdict(int)
        for label_pair_idx, label in enumerate(label_tagsnameset_d.keys()):
            reoccurent_tagnames = tagnames_set.intersection(label_tagsnameset_d[label])
            for tagname in list(reoccurent_tagnames):
                tagnames_reoccurentcount_d[tagname] += 1
        reoccurentcounts_l = sorted([reoccurentcount for reoccurentcount in tagnames_reoccurentcount_d.values()], reverse=True)
        # reoccurentcounts_l_normalized = [round(reoccurentcounts_l_entry/sum(reoccurentcounts_l)*100, 2) for reoccurentcounts_l_entry in reoccurentcounts_l]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=600)
        # proba_array = proba_array.reshape(-1)
        # c_l = [color_l[cluster_idx] for cluster_idx in yhats]
        bar_plots = ax.bar(list(range(len(reoccurentcounts_l))), reoccurentcounts_l)
        ax.bar_label(bar_plots)
        # ax.set_xlim(-1, 5.5)
        ax.set_xticks(list(range(len(reoccurentcounts_l))))
        # ax.set_title('Probability Plot', fontdict={'fontsize': 30, 'fontweight': 'medium'})
        ax.set_xlabel("Tokens", fontdict={'fontsize': 26})
        ax.set_ylabel("Number of Packages", fontdict={'fontsize': 26})
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        # ax.bar_label(bar_plots, labels=yhats, fontsize=10)
        # ax.vlines(x=biggest_yhat_idx-0.5, ymin=min(proba_array), ymax=max(proba_array), color='black')
        plt.savefig(dirname+plot_dir+target_dir+'_distribution_tagsreoccurentinpackagepair.pdf', bbox_inches='tight')
        plt.close()

        # plot count of token occurences
        reoccurentcounts_counter = Counter(reoccurentcounts_l)
        # reoccurentcounts_value_l = list(reoccurentcounts_counter.keys())
        reoccurentcounts_valuecount_l = sorted(list(reoccurentcounts_counter.items()), key=lambda x: x[1],reverse=True)
        reoccurentcounts_value_l = [str(value) for value, _ in reoccurentcounts_valuecount_l]
        reoccurentcounts_valuecount_l = [count for _, count in reoccurentcounts_valuecount_l]
        reoccurentcounts_valuecount_sum = sum(reoccurentcounts_valuecount_l)
        reoccurentcounts_valuecount_l_nomalized = [round(reoccurentcounts_valuecount/reoccurentcounts_valuecount_sum*100,2) for reoccurentcounts_valuecount in reoccurentcounts_valuecount_l]
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        # proba_array = proba_array.reshape(-1)
        # c_l = [color_l[cluster_idx] for cluster_idx in yhats]
        bar_plots = ax.bar(list(range(len(reoccurentcounts_valuecount_l_nomalized))), reoccurentcounts_valuecount_l_nomalized, hatch="*")
        ax.bar_label(bar_plots)
        ax.grid()
        ax.set_xticks(list(range(len(reoccurentcounts_valuecount_l_nomalized))))
        ax.set_xticklabels(reoccurentcounts_value_l)
        ax.set_xlim(-1, 5.5)
        # ax.set_title('Probability Plot', fontdict={'fontsize': 30, 'fontweight': 'medium'})
        ax.set_xlabel("Number of Packages", fontdict={'fontsize': 20})
        ax.set_ylabel("% of tokens", fontdict={'fontsize': 20})
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        # ax.bar_label(bar_plots, labels=yhats, fontsize=10)
        # ax.vlines(x=biggest_yhat_idx-0.5, ymin=min(proba_array), ymax=max(proba_array), color='black')
        plt.savefig(dirname+plot_dir+target_dir+'_distribution_countoftokenoccurences.pdf', bbox_inches='tight')
        plt.close()


    return

if __name__ == '__main__':
    plot_size()