import os
import sys
import time
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import yaml, json, statistics
from tqdm import tqdm
from collections import defaultdict, Counter
from pathlib import Path
import multiprocessing as mp
import itertools
import matplotlib
import xgboost as xgb
import pickle
import numpy as np
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

def get_intersection(from_set, to_set):
    ret = []
    for ele in from_set:
        if ele in to_set:
            ret.append(ele)
    return ret

def plot_size():
    # tagset
    # sizes_l, p_l = [], []
    combination_length_d = defaultdict(int)
    pair_count_d = defaultdict(int)
    pair_tag_count_d = defaultdict(list)
    label_tags_d = defaultdict(lambda: defaultdict(list))
    # label_pairs_l = list()
    # tagnames_set = set()
    label_tagsnameset_d = defaultdict(set)
    target_dir = "tagset_ML_3_test"
    dirname = "/home/cc/Praxi-Pipeline/data/data_4/"
    out_dirname = dirname+target_dir+"/"
    filter_dir = "filters/"
    plot_dir = "plots-1/"
    Path(dirname+filter_dir).mkdir(parents=True, exist_ok=True)
    Path(dirname+plot_dir).mkdir(parents=True, exist_ok=True)

    # print(out_dirname)
    tagsetfilenames_l = [name for name in os.listdir(out_dirname) if os.path.isfile(out_dirname+name) and name[-4:]!=".obj"]
    # print(tagsetfilenames_l)
    # if len(tagsetfilenames_l) == 2:
    # pool = mp.Pool(processes=mp.cpu_count())
    pool = mp.Pool(processes=16)
    data_instance_d_l = [pool.apply_async(read_tagset, args=(out_dirname+tagsets_name,)) for tagsets_name in tqdm(tagsetfilenames_l)]
    data_instance_d_l = [data_instance_d.get() for data_instance_d in tqdm(data_instance_d_l) if data_instance_d.get()!=None]
    pool.close()
    pool.join()
    for data_instance_d in data_instance_d_l:
        combination_length_d[data_instance_d["label_count"]] += 1
        pair_count_d[data_instance_d["labels_str"]] += 1
        pair_tag_count_d[data_instance_d["labels_str"]].append(data_instance_d["tags_length"])
        # label_pairs_l.append(data_instance_d["labels_str"])
        for tagname, tagoccurence in data_instance_d["tags_d"].items():
            label_tags_d[data_instance_d["labels_str"]][tagname].append(tagoccurence)   # {"label":{"tagname":tagoccurence,},}
        # tagnames_set.update(set(data_instance_d["tags_d"].keys()))                      # all tagname; {"tagname",}
        label_tagsnameset_d[data_instance_d["labels_str"]].update(set(data_instance_d["tags_d"].keys())) # {"label":{"tagname",},}
    # for tagsets_name in tqdm(tagsetfilenames_l):
    #     # print(out_dirname+tagsets_name)
        
    #     with open(out_dirname+tagsets_name, "r") as stream:
    #         try:
    #             tagset = yaml.safe_load(stream)
    #             if 'labels' in tagset:
    #                 label_count = len(tagset['labels'])
    #                 combination_length_d[str(label_count)] += 1

    #                 labels_str = '&&'.join(tagset['labels'])
    #                 pair_count_d[labels_str] += 1
    #                 pair_tag_count_d[labels_str].append(len(tagset['tags']))
    #             else:
    #                 combination_length_d[str(1)] += 1
    #                 pair_count_d[tagset['label']] += 1
    #                 pair_tag_count_d[labels_str].append(len(tagset['tags']))
    #         except yaml.YAMLError as exc:
    #             print(exc)



    # models
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
    cwd_clf = "/home/cc/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts_bak"

    clf_path_l = []
    for i in range(n_models):
        clf_pathname = f"{cwd_clf}/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(clf_njobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/model_init.json"
        if os.path.isfile(clf_pathname):
            clf_path_l.append(clf_pathname)
        else:
            print(f"clf is missing: {clf_pathname}")
            sys.exit(-1)

    clf_tagsnameset_d = {}
    for clf_idx, clf_path in enumerate(clf_path_l):
        BOW_XGB = xgb.XGBClassifier(max_depth=10, learning_rate=0.1,silent=False, objective='binary:logistic', \
                        booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                        subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0, reg_lambda=1)
        BOW_XGB.load_model(clf_path)
        feature_importance = BOW_XGB.feature_importances_
        index_tag_mapping_path=clf_path[:-15]+'index_tag_mapping'
        with open(index_tag_mapping_path, 'rb') as fp:
            all_tags_l = pickle.load(fp)
        used_tags_set = set([all_tags_l[used_fidx] for used_fidx in np.where(feature_importance > 0)[0].tolist()])
        clf_tagsnameset_d[clf_path] = used_tags_set


    clf_usage_d = defaultdict(int)
    for clf, tagsnamset in clf_tagsnameset_d.items():
        for data_instance_d in data_instance_d_l:
            used_instance_tags_list = get_intersection(tagsnamset, set(data_instance_d["tags_d"].keys()))
            if used_instance_tags_list:
                clf_usage_d[clf] += 1
    clf_usage_d = {k: v for k, v in sorted(clf_usage_d.items(), key=lambda item: item[1])}
    with open(dirname+plot_dir+target_dir+"_clf_usage_d", 'w') as outfile:
        yaml.dump(clf_usage_d, outfile, default_flow_style=False)


    pair_count_d = {k: v for k, v in sorted(pair_count_d.items(), key=lambda item: item[1])}
    with open(dirname+plot_dir+target_dir+"_pair_count_d", 'w') as outfile:
        yaml.dump(pair_count_d, outfile, default_flow_style=False)
    fig, ax = plt.subplots(1, 1, figsize=(26, 6), dpi=600)
    # proba_array = proba_array.reshape(-1)
    # c_l = [color_l[cluster_idx] for cluster_idx in yhats]
    bar_plots = ax.bar(list(range(len(pair_count_d))), list(pair_count_d.values()))
    ax.bar_label(bar_plots)
    # ax.set_xlim(-2, len(proba_array)+1)
    ax.set_xticks(list(range(len(pair_count_d))))
    ax.set_xticklabels(list(pair_count_d.keys()), rotation=90)
    # ax.set_title('Probability Plot', fontdict={'fontsize': 30, 'fontweight': 'medium'})
    ax.set_xlabel(str(len(pair_count_d))+" label idx", fontdict={'fontsize': 26})
    ax.set_ylabel("Tags Count", fontdict={'fontsize': 26})
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    # ax.bar_label(bar_plots, labels=yhats, fontsize=10)
    # ax.vlines(x=biggest_yhat_idx-0.5, ymin=min(proba_array), ymax=max(proba_array), color='black')
    plt.savefig(dirname+plot_dir+target_dir+'_distribution_tags_count.pdf', bbox_inches='tight')
    plt.close()

    combination_length_d = {k: v for k, v in sorted(combination_length_d.items(), key=lambda item: item[1])}
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=600)
    # proba_array = proba_array.reshape(-1)
    # c_l = [color_l[cluster_idx] for cluster_idx in yhats]
    bar_plots = ax.bar(list(range(len(combination_length_d))), list(combination_length_d.values()))
    ax.bar_label(bar_plots)
    # ax.set_xlim(-2, len(proba_array)+1)
    ax.set_xticks(list(range(len(combination_length_d))))
    ax.set_xticklabels(list(combination_length_d.keys()), rotation=90)
    # ax.set_title('Probability Plot', fontdict={'fontsize': 30, 'fontweight': 'medium'})
    ax.set_xlabel("Combination Length", fontdict={'fontsize': 26})
    ax.set_ylabel("Combination Length Count", fontdict={'fontsize': 26})
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    # ax.bar_label(bar_plots, labels=yhats, fontsize=10)
    # ax.vlines(x=biggest_yhat_idx-0.5, ymin=min(proba_array), ymax=max(proba_array), color='black')
    plt.savefig(dirname+plot_dir+target_dir+'_distribution_combination_length.pdf', bbox_inches='tight')
    plt.close()


    # Plot clf usage for each clf
    # clf_usage_d = {0: 10, 1: 15, 2: 7, ...}
    # Get classifier indices and corresponding usage counts
    # clf_idxs = sorted(clf_usage_d.keys())  # ensures the x-axis values are ordered
    # usage_counts = [clf_usage_d[idx] for idx in clf_idxs]
    # clf_usage_d = {k: v for k, v in sorted(clf_usage_d.items(), key=lambda item: item[1])}

    # Create a bar plot
    plt.figure(figsize=(10, 6))  # optionally adjust the figure size
    # plt.bar(clf_idxs, usage_counts, color='skyblue')
    plt.bar(list(range(len(clf_usage_d))), list(clf_usage_d.values()), color='skyblue')
    plt.xlabel('Classifier Index (clf_idx)')
    plt.ylabel('Usage Count')
    plt.title('Classifier Usage by clf_idx')
    # plt.xticks(list(range(len(clf_usage_d))))  # ensure each classifier is marked on the x-axis
    plt.savefig(dirname+plot_dir+target_dir+'_distribution_clf_usage.pdf', bbox_inches='tight')

    # # Sort the dictionary by usage count in ascending order.
    # sorted_clf_usage = dict(sorted(clf_usage_d.items(), key=lambda item: item[1]))
    # Extract classifier indices and the corresponding sorted usage counts.
    clf_idxs = list(range(len(clf_usage_d)))
    usage_counts = list(clf_usage_d.values())
    # Compute the cumulative sum of the usage counts.
    cdf_values = np.cumsum(usage_counts)
    # Normalize the cumulative values to obtain a probability CDF (ranging from 0 to 1).
    cdf_normalized = cdf_values / cdf_values[-1]
    plt.figure(figsize=(10, 6))
    # Plot the normalized (probability) CDF using a step plot.
    plt.step(clf_idxs, cdf_normalized, where='mid', color='skyblue', label='Normalized CDF')
    # If you prefer to plot the absolute cumulative counts, uncomment the next lines and comment out the normalized plot.
    # plt.step(clf_idxs, cdf_values, where='mid', color='skyblue', label='Cumulative Usage Count')
    # plt.ylabel('Cumulative Usage Count')
    plt.xlabel('Classifier Index (clf_idx)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Classifier Usage')
    plt.legend()
    plt.grid(True)
    # Save the plot as a PDF file.
    plt.savefig(dirname + plot_dir + target_dir + '_cdf_clf_usage.pdf', bbox_inches='tight')
    # plt.show()

    # ######################## Plots below are for SL datasets only
    # Plot tag counts for each label pair
    # highlight_label_l = ['platformdirs', 'certifi', 'jmespath', 'aiohttp', 'async-timeout', 'pyparsing', 'pydantic', 'importlib-resources', 'websocket-client', 'aiosignal', 'distlib', 'gitpython', 'tabulate', 'proto-plus', 'msal', 'azure-storage-blob', 'tzlocal', 'docker', 'grpcio-tools', 'sqlparse', 'wcwidth', 'poetry-core', 'sniffio', 'google-auth-oauthlib', 'jaraco-classes', 'dill', 'alembic', 'httplib2', 'python-dotenv', 'scramp', 'tb-nightly', 'marshmallow', 'uritemplate', 'toml', 'trove-classifiers', 'cycler', 'jeepney', 'pyzmq', 'toolz', 'prometheus-client', 'httpcore', 'adal', 'shellingham', 'pyflakes', 'httpx', 'pkginfo', 'sentry-sdk', 'nbconvert', 'fastapi', 'flake8', 'python-utils', 'asynctest', 'google-cloud-bigquery-storage', 'databricks-cli', 'starlette', 'aioitertools', 'pickleshare', 'mistune', 'jupyter-server', 'pbr', 'ipykernel', 'build', 'arrow', 'asgiref', 'uvicorn', 'html5lib', 'pyproject-hooks', 'oauth2client', 'tinycss2', 'altair', 'multiprocess', 'zope-interface', 'retry', 'crashtest', 'httptools', 'querystring-parser', 'contextlib2', 'tensorboard-data-server', 'azure-storage-file-datalake', 'xlsxwriter', 'configparser', 'mysql-connector-python', 'pendulum', 'text-unidecode', 'semver', 'responses', 'pipenv', 'snowflake-sqlalchemy', 'python-slugify', 'pytest-xdist', 'sphinx', 'jupyterlab-widgets', 'gremlinpython', 'click-plugins', 'pytest-mock', 'azure-storage-common', 'dataclasses-json', 'futures', 'pandocfilters', 'patsy', 'xxhash', 'tensorflow-io-gcs-filesystem', 'jupyterlab-pygments', 'setproctitle', 'astunparse', 'async-lru', 'gcsfs', 'azure-keyvault-secrets', 'pysftp', 'ordered-set', 'faker', 'semantic-version', 'jsonpickle', 'pytest-runner', 'sphinxcontrib-serializinghtml', 'webcolors', 'azure-datalake-store', 'typing', 'isoduration', 'jupyter-server-terminals', 'deprecation', 'opencensus-context', 'typed-ast', 'opencensus', 'stevedore', 'pyproj', 'gspread', 'ppft', 'watchtower', 'trio-websocket', 'azure-mgmt-keyvault', 'structlog', 'opentelemetry-exporter-otlp-proto-http', 'opentelemetry-semantic-conventions', 'enum34', 'pathlib2', 'types-urllib3', 'pybind11', 'pydata-google-auth', 'lightgbm', 'opencensus-ext-azure', 'lz4', 'cligj', 'azure-mgmt-containerregistry', 'keras-preprocessing', 'unittest-xml-reporting', 'partd', 'schema', 'flask-cors', 'alabaster', 'azure-mgmt-authorization', 'h2', 'python-http-client', 'amqp', 'pytest-asyncio', 'locket', 'hyperframe']
    # highlight_label_l = ['s3fs', 'cryptography', 'emoji']
    # highlight_label_l = []
    pair_tag_count_d = {k: sum(v)/len(v) for k, v in pair_tag_count_d.items()}
    pair_tag_count_d = {k: v for k, v in sorted(pair_tag_count_d.items(), key=lambda item: item[1])}
    pair_tag_count_ktoidx = {}
    for idx, pair in enumerate(pair_tag_count_d):
        pair_tag_count_ktoidx[pair] = idx
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=600)
    # proba_array = proba_array.reshape(-1)
    # c_l = [color_l[cluster_idx] for cluster_idx in yhats]
    bar_plots = ax.bar(list(range(len(pair_tag_count_d))), list(pair_tag_count_d.values()))
    # ax.bar_label(bar_plots)
    # ax.vlines([pair_tag_count_ktoidx[highlight_label] for highlight_label in highlight_label_l],ymin=[0 for _ in highlight_label_l],ymax=[pair_tag_count_d[pair] for pair in highlight_label_l],colors="r")
    # ax.set_xlim(-2, len(proba_array)+1)
    ax.set_xticks(list(range(len(pair_tag_count_d))))
    # ax.set_title('Probability Plot', fontdict={'fontsize': 30, 'fontweight': 'medium'})
    ax.set_xlabel("Packages", fontdict={'fontsize': 26})
    ax.set_ylabel("Number of Tokens", fontdict={'fontsize': 26})
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    # ax.bar_label(bar_plots, labels=yhats, fontsize=10)
    # ax.vlines(x=biggest_yhat_idx-0.5, ymin=min(proba_array), ymax=max(proba_array), color='black')
    plt.savefig(dirname+plot_dir+target_dir+'_distribution_tagsofpackagepair.pdf', bbox_inches='tight')
    plt.close()


    # # Plot tag reoccurence in all tagsets.
    # highlight_labeltagnames_set = set()
    # for highlight_label in highlight_label_l:
    #     highlight_labeltagnames_set.update(label_tagsnameset_d[highlight_label])
    # highlight_labeltagnames_reoccurentlabel_d = defaultdict(list)
    # highlight_labeltagnames_reoccurentcount_d = defaultdict(int)
    # for label_pair_idx, label in enumerate(label_tagsnameset_d.keys()):
    #     reoccurent_tagnames = highlight_labeltagnames_set.intersection(label_tagsnameset_d[label])
    #     for tagname in list(reoccurent_tagnames):
    #         highlight_labeltagnames_reoccurentcount_d[tagname] += 1
    #         highlight_labeltagnames_reoccurentlabel_d[tagname].append(label)
    # # print(sorted([(tagname, reoccurentcount) for tagname, reoccurentcount in highlight_labeltagnames_reoccurentcount_d.items() if reoccurentcount>1],key=lambda x: x[1], reverse=True))
    # tokens_filter_l = sorted([tagname for tagname, reoccurentcount in highlight_labeltagnames_reoccurentcount_d.items() if reoccurentcount>1],key=lambda x: x[1], reverse=True)
    # # print(tokens_filter_l)
    # tokens_filter_set = set(tokens_filter_l)
    # with open(dirname+filter_dir+target_dir+"_tokenshares_filter_set", 'w') as f:
    #     yaml.dump(tokens_filter_set, f)
    #     # for s in tokens_filter_l:
    #     #     f.write(s + '\n')
    # # print(json.dumps(highlight_labeltagnames_reoccurentcount_d,sort_keys=True, indent=4))
    # tagname_morethan1occurence_set = set([tagname for tagname, reoccurentcount in highlight_labeltagnames_reoccurentcount_d.items() if reoccurentcount>1])
    # label_tagsnameset_d_dropreoccurenttagnames = {}
    # for label, tagsnameset in label_tagsnameset_d.items():
    #     label_tagsnameset_d_dropreoccurenttagnames[label] = tagsnameset-tagname_morethan1occurence_set
    # # print(sorted([(label, tagsnameset) for label, tagsnameset in label_tagsnameset_d_dropreoccurenttagnames.items() if len(tagsnameset)<1],key=lambda x: x[1], reverse=True))
    # # print(json.dumps(label_tagsnameset_d_dropreoccurenttagnames,sort_keys=True, indent=4))
    # # print()

    tagnames_set = set()
    for label in label_tagsnameset_d.keys():
        tagnames_set.update(label_tagsnameset_d[label])
    tagnames_reoccurentcount_d = defaultdict(int)
    for label_pair_idx, label in enumerate(label_tagsnameset_d.keys()):
        reoccurent_tagnames = tagnames_set.intersection(label_tagsnameset_d[label])
        for tagname in list(reoccurent_tagnames):
            tagnames_reoccurentcount_d[tagname] += 1
    with open(dirname+filter_dir+target_dir+"_tagnames_reoccurentcount_d", "w") as outfile:
        yaml.dump(tagnames_reoccurentcount_d, outfile)
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
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

    # tagname_morethan1occurence_set = set()
    # tagname_reoccurentcount_d = defaultdict(int)
    # for (highlight_label, label) in itertools.product(highlight_label_l, label_pairs_l):
    #     if label != highlight_label:
    #         reoccurent_tagnames = label_tagsnameset_d[highlight_label].intersection(label_tagsnameset_d[label])
    #         tagname_morethan1occurence_set.update(reoccurent_tagnames)
    #         for tagname in list(reoccurent_tagnames):
    #             tagname_reoccurentcount_d[tagname] += 1
    # label_tagsnameset_d_dropreoccurenttagnames = {}
    # for label, tagsnameset in label_tagsnameset_d.items():
    #     label_tagsnameset_d_dropreoccurenttagnames[label] = tagsnameset-tagname_morethan1occurence_set
    # print(len([1 for tagnames_set in label_tagsnameset_d_dropreoccurenttagnames.values() if len(tagnames_set)==0]))
    # print(tagname_morethan1occurence_set)
    # print(json.dumps(tagname_reoccurentcount_d,sort_keys=True, indent=4))
    # print(json.dumps(label_tagsnameset_d_dropreoccurenttagnames,sort_keys=True, indent=4))
    # print()


    # # Plot random noise tags in each labeltagset
    # label_tagnames_reoccurentcount_d = defaultdict(lambda: defaultdict(int))
    # tagnamesbelowoccurence_set = set()
    # for label, tags_d in label_tags_d.items():
    #     for tagname, counts_l in tags_d.items():
    #         label_tagnames_reoccurentcount_d[label][tagname] += len(counts_l)
    #         if len(counts_l) < 5:
    #             tagnamesbelowoccurence_set.add(tagname)
    
    # # print()
    # # print(tagnamesbelowoccurence_set)
    # with open(dirname+filter_dir+target_dir+"_tokennoises_filter_set", 'w') as f:
    #     yaml.dump(tagnamesbelowoccurence_set, f)
    #     # for s in list(tagnamesbelowoccurence_set):
    #     #     f.write(s + '\n')

    # # for tagsnameset_idx, (label, tagsname_set) in enumerate(label_tagsnameset_d.items()):
    # #     for tagsname in list(tagsname_set):
    # #         tagnames_reoccurentcount_d[label][tagsname] += 1


    # # plot count of token occurences after filter shared token
    # reoccurentcounts_l = sorted([reoccurentcount for tagname, reoccurentcount in tagnames_reoccurentcount_d.items() if tagname not in tokens_filter_set], reverse=True)
    # reoccurentcounts_counter = Counter(reoccurentcounts_l)
    # # reoccurentcounts_value_l = list(reoccurentcounts_counter.keys())
    # reoccurentcounts_valuecount_l = sorted(list(reoccurentcounts_counter.values()),reverse=True)
    # reoccurentcounts_valuecount_sum = sum(reoccurentcounts_valuecount_l)
    # reoccurentcounts_valuecount_l_nomalized = [round(reoccurentcounts_valuecount/reoccurentcounts_valuecount_sum*100,2) for reoccurentcounts_valuecount in reoccurentcounts_valuecount_l]
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=600)
    # # proba_array = proba_array.reshape(-1)
    # # c_l = [color_l[cluster_idx] for cluster_idx in yhats]
    # bar_plots = ax.bar(list(range(len(reoccurentcounts_valuecount_l_nomalized))), reoccurentcounts_valuecount_l_nomalized, hatch="*")
    # ax.bar_label(bar_plots)
    # ax.set_xlim(-1, 5.5)
    # ax.grid()
    # ax.set_xticks([])
    # # ax.set_title('Probability Plot', fontdict={'fontsize': 30, 'fontweight': 'medium'})
    # ax.set_xlabel("Count", fontdict={'fontsize': 26})
    # ax.set_ylabel("% of Tokens", fontdict={'fontsize': 26})
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.tick_params(axis='both', which='minor', labelsize=10)
    # # ax.bar_label(bar_plots, labels=yhats, fontsize=10)
    # # ax.vlines(x=biggest_yhat_idx-0.5, ymin=min(proba_array), ymax=max(proba_array), color='black')
    # plt.savefig(dirname+plot_dir+target_dir+'_distribution_countoftokenoccurencesafterfiltersharedtoken.pdf', bbox_inches='tight')
    # plt.close()

    # # plot count of token occurences after filter noise and shared token
    # filternoiseandsharedtoken = tokens_filter_set.union(tagnamesbelowoccurence_set)
    # reoccurentcounts_l = sorted([reoccurentcount for tagname, reoccurentcount in tagnames_reoccurentcount_d.items() if tagname not in filternoiseandsharedtoken], reverse=True)
    # reoccurentcounts_counter = Counter(reoccurentcounts_l)
    # # reoccurentcounts_value_l = list(reoccurentcounts_counter.keys())
    # reoccurentcounts_valuecount_l = sorted(list(reoccurentcounts_counter.values()),reverse=True)
    # reoccurentcounts_valuecount_sum = sum(reoccurentcounts_valuecount_l)
    # reoccurentcounts_valuecount_l_nomalized = [round(reoccurentcounts_valuecount/reoccurentcounts_valuecount_sum*100,2) for reoccurentcounts_valuecount in reoccurentcounts_valuecount_l]
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=600)
    # # proba_array = proba_array.reshape(-1)
    # # c_l = [color_l[cluster_idx] for cluster_idx in yhats]
    # bar_plots = ax.bar(list(range(len(reoccurentcounts_valuecount_l_nomalized))), reoccurentcounts_valuecount_l_nomalized, hatch="*")
    # ax.bar_label(bar_plots)
    # ax.set_xlim(-1, 5.5)
    # ax.grid()
    # ax.set_xticks([])
    # # ax.set_title('Probability Plot', fontdict={'fontsize': 30, 'fontweight': 'medium'})
    # ax.set_xlabel("Count", fontdict={'fontsize': 26})
    # ax.set_ylabel("% of Tokens", fontdict={'fontsize': 26})
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.tick_params(axis='both', which='minor', labelsize=10)
    # # ax.bar_label(bar_plots, labels=yhats, fontsize=10)
    # # ax.vlines(x=biggest_yhat_idx-0.5, ymin=min(proba_array), ymax=max(proba_array), color='black')
    # plt.savefig(dirname+plot_dir+target_dir+'_distribution_countoftokenoccurencesafterfilternoiseandsharedtoken.pdf', bbox_inches='tight')
    # plt.close()




    # filtered_pair_tagsetfilenames_d = defaultdict(list)
    # filtered_tagsetfilenames_set = set()
    # for instance_idx, data_instance_d in enumerate(data_instance_d_l):
    #     if len(data_instance_d["tagnames_set"] - filternoiseandsharedtoken) > 0:
    #         filtered_pair_tagsetfilenames_d[data_instance_d["labels_str"]].append(tagsetfilenames_l[instance_idx])
    #         filtered_tagsetfilenames_set.add(tagsetfilenames_l[instance_idx])
    # with open(dirname+filter_dir+target_dir+"_filtered_tagsets_set", 'w') as f:
    #     yaml.dump(filtered_tagsetfilenames_set, f)

    # max_tagsetsperpairlength = 25
    # for k,v in filtered_pair_tagsetfilenames_d.items():
    #     # print(k, len(v))
    #     if len(v) < max_tagsetsperpairlength:
    #         max_tagsetsperpairlength = len(v)
    # filtered_tagsets_set_same_length = set()
    # filtered_pair_tagsetfilenames_d_same_length = defaultdict(list)
    # for k,v in filtered_pair_tagsetfilenames_d.items():
    #     filtered_tagsets_set_same_length.update(v[:max_tagsetsperpairlength])
    #     filtered_pair_tagsetfilenames_d_same_length[k] = v[:max_tagsetsperpairlength]
    # with open(dirname+filter_dir+target_dir+"_filtered_tagsets_set_same_length", 'w') as f:
    #     yaml.dump(filtered_tagsets_set_same_length, f)
    # with open(dirname+filter_dir+target_dir+"_filtered_pair_tagsetfilenames_d_same_length", 'w') as f:
    #     yaml.dump(filtered_pair_tagsetfilenames_d_same_length, f)
    

    # # with open(dirname+filter_dir+"tagset_files.yaml", 'rb') as tf:
    # #     test_filtered_tagsets_set = yaml.load(tf, Loader=yaml.Loader)


    return

if __name__ == '__main__':
    plot_size()