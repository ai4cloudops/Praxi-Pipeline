import os
from collections import defaultdict
import json, pickle

# # Read all data
# # # data 0 experiments
# data_d = dict()
# n_samples = 25
# for dataset in ["data_0"]:
#     data_dir = "data0_results/"
#     for with_filter in [False]:
#         for n_jobs in [1]:
#             for clf_njobs in [8]:
#                 for n_models, test_batch_count in zip([8,4,2,1],[1,1,1,1]): #([50,25,20,15,10,5,1],[1,1,1,1,1,1,8]): # ([1,25,10],[8,1,1])
#                     data_d[str(n_models)+"_models"] = dict()
#                     for n_estimators in [100]:
#                         data_d[str(n_models)+"_models"][n_estimators] = dict()
#                         for depth in [1]:
#                             for tree_method in["exact"]: # "exact","approx","hist"
#                                 for max_bin in [1]:
                                    
#                                     for input_size, dim_compact_factor in zip([None, 13664, 27329, 54659, 109319],[1,1,1,1,1]): # [None, 10000, 1000, 500, 5000, 15000],[1,1,1,1,1,1] # [None, 13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319] [50, 100, 250, 6832, 18000, 27329, 60000, 109319]
#                                         data_d[str(n_models)+"_models"][n_estimators][input_size] = defaultdict(list)
                                        
#                                         for shuffle_idx in range(3):
#                                             for test_sample_batch_idx in [4,1,2,0,3]:
#                                                 traintime_pertrail_l = []
#                                                 for i in range(n_models):
#                                                     filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+"removesharedornoisestags/metrics_init.out"
#                                                     if os.path.isfile(filename):
#                                                         with open(filename) as file:
#                                                             for line in file:
#                                                                 line_l = line.rstrip().split(":")
#                                                                 if line_l[0] == "BOW_XGB_init.fit":
#                                                                     traintime_pertrail_l.append(float(line_l[1]))
#                                                                     # data_d[str(n_models)+"_models"][n_estimators][input_size]["traintime"].append(float(line_l[1]))
#                                                                 if line_l[0] == "tagsets_to_matrix-trainset_xsize":
#                                                                     data_d[str(n_models)+"_models"][n_estimators][input_size]["samplesize"].append(float(line_l[1]))
#                                                                 if line_l[0] == "tagsets_to_matrix-trainset_ysize":
#                                                                     data_d[str(n_models)+"_models"][n_estimators][input_size]["dimensions"].append(float(line_l[1]))
#                                                                 if line_l[0] == "F1 SCORE ":
#                                                                     data_d[str(n_models)+"_models"][n_estimators][input_size]["CV-f1-score"].append(float(line_l[1].split(" ", 2)[1]))
#                                                     else:
#                                                         print(filename)
#                                                 data_d[str(n_models)+"_models"][n_estimators][input_size]["traintime"].append(traintime_pertrail_l)
#                                         for shuffle_idx in range(3):
#                                             for test_sample_batch_idx in [0,1,2,3,4]:
#                                                 # cwd =      "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"_shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_removesharedornoisestags/"
#                                                 filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+"removesharedornoisestags/metrics_pred.out"
#                                                 if os.path.isfile(filename):
#                                                     try:
#                                                         with open(filename, "r") as f:
#                                                             inferencetime_in_a_bag = 0
#                                                             for line in f:
#                                                                 if line[:len(" BOW_XGB.predict_")] == " BOW_XGB.predict_":
#                                                                     line_l = line.split(":")
#                                                                     inferencetime_in_a_bag += round(float(line_l[1]),2)
#                                                                 else:
#                                                                     line_l = line.rstrip().split()
#                                                                     if len(line_l) == 9:
#                                                                         data_d[str(n_models)+"_models"][n_estimators][input_size]["f1-score"].append((float(line_l[3])))
#                                                                         # data_d[str(n_models)+"_models"][n_estimators][input_size]["f1-score-config"].append(str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx")
#                                                                     if len(line_l) == 7:
#                                                                         data_d[str(n_models)+"_models"][n_estimators][input_size]["precision"].append((float(line_l[1])))
#                                                                     if len(line_l) == 8:
#                                                                         data_d[str(n_models)+"_models"][n_estimators][input_size]["recall"].append((float(line_l[2])))
#                                                             data_d[str(n_models)+"_models"][n_estimators][input_size]["inferencetime"].append(round(inferencetime_in_a_bag,2))
#                                                     except Exception as e:
#                                                         print(e)
#                                                 else:
#                                                     print(filename)

                                                    
# for nmodel,datadict in data_d.items():
#     print(nmodel)
#     for metric, value in datadict.items():
#         # print(metric, value)
#         print(metric, value)
#         print()
#     print()







# # Read all data
# # # data 3 experiments
# data_d = dict()
# n_samples = 21
# for dataset in ["data_3"]:
#     data_dir = "data3_results/"
#     for with_filter in ["True"]:
#         for n_jobs in [32]:
#             for clf_njobs in [32]:
#                 for n_models, test_batch_count in zip([50,25,20,15,10,5,1],[1,1,1,1,1]): #([50,25,20,15,10,5,1],[1,1,1,1,1,1,8]): # ([1,25,10],[8,1,1])
#                     data_d[str(n_models)+"_models"] = dict()
#                     for n_estimators in [100]: # 10,50,100
#                         data_d[str(n_models)+"_models"][n_estimators] = dict()
#                         for depth in [1]:
#                             for tree_method in["exact"]: # "exact","approx","hist"
#                                 for max_bin in [1]:
                                    
#                                     for input_size, dim_compact_factor in zip([None],[1]): # [None, 10000, 1000, 500, 5000, 15000],[1,1,1,1,1,1] # [None, 13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
#                                         data_d[str(n_models)+"_models"][n_estimators][input_size] = defaultdict(list)
#                                         for i in range(n_models):
#                                             for shuffle_idx in range(3):
#                                                 for test_sample_batch_idx in [0,1,2,3,4]:
#                                                     filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+"removesharedornoisestags/metrics_init.out"
#                                                     if os.path.isfile(filename):
#                                                         with open(filename) as file:
#                                                             for line in file:
#                                                                 line_l = line.rstrip().split(":")
#                                                                 if line_l[0] == "BOW_XGB_init.fit":
#                                                                     data_d[str(n_models)+"_models"][n_estimators][input_size]["traintime"].append(float(line_l[1]))
#                                                                 if line_l[0] == "tagsets_to_matrix-trainset_xsize":
#                                                                     data_d[str(n_models)+"_models"][n_estimators][input_size]["samplesize"].append(float(line_l[1]))
#                                                                 if line_l[0] == "tagsets_to_matrix-trainset_ysize":
#                                                                     data_d[str(n_models)+"_models"][n_estimators][input_size]["dimensions"].append(float(line_l[1]))
#                                                     else:
#                                                         print(filename)
#                                         for shuffle_idx in range(3):
#                                             for test_sample_batch_idx in [0,1,2,3,4]:
#                                                 # cwd =      "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"_shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_removesharedornoisestags/"
#                                                 filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+"removesharedornoisestags/metrics_pred.out"
#                                                 if os.path.isfile(filename):
#                                                     try:
#                                                         with open(filename, "r") as f:
#                                                             inferencetime_in_a_bag = 0
#                                                             for line in f:
#                                                                 if line[:len(" BOW_XGB.predict_")] == " BOW_XGB.predict_":
#                                                                     line_l = line.split(":")
#                                                                     inferencetime_in_a_bag += round(float(line_l[1]),2)
#                                                                 else:
#                                                                     line_l = line.rstrip().split()
#                                                                     if len(line_l) == 9:
#                                                                         data_d[str(n_models)+"_models"][n_estimators][input_size]["f1-score"].append((float(line_l[3])))
#                                                                         # data_d[str(n_models)+"_models"][n_estimators][input_size]["f1-score-config"].append(str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx")
#                                                                     if len(line_l) == 7:
#                                                                         data_d[str(n_models)+"_models"][n_estimators][input_size]["precision"].append((float(line_l[1])))
#                                                                     if len(line_l) == 8:
#                                                                         data_d[str(n_models)+"_models"][n_estimators][input_size]["recall"].append((float(line_l[2])))
#                                                             data_d[str(n_models)+"_models"][n_estimators][input_size]["inferencetime"].append(round(inferencetime_in_a_bag,2))
#                                                     except Exception as e:
#                                                         print(e)
#                                                 else:
#                                                     print(filename)

                                                    
# for nmodel,datadict in data_d.items():
#     print(nmodel)
#     for metric, value in datadict.items():
#         print(metric, value)
#         print()
#     print()









# Read all data
# # data 4 experiments
data_d = dict()
n_samples = 4
for dataset in ["data_4"]:
    data_dir = ""
    for (with_filter, freq) in [[(False, 100),(True, 25)][0]]:
        for n_jobs in [32]:
            for clf_njobs in [32]:
                # for n_models, sim_thr, test_batch_count in zip([1000, 500],[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.7],[1,1,1,1,1,1,1]):
                for n_models, sim_thr, test_batch_count in zip([25, 25, 25, 25, 25, 25],[0.98, 0.9, 0.8, 0.7, 0.6],[1,1,1,1,1,1,1,1,1,1]):
                    data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"] = dict()
                    for n_estimators in [100]: # 10,50,100
                        data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators] = dict()
                        for depth in [1]:
                            for tree_method in["exact"]: # "exact","approx","hist"
                                for max_bin in [1]:
                                    
                                    for input_size, dim_compact_factor in zip([None],[1]): # [None, 10000, 1000, 500, 5000, 15000],[1,1,1,1,1,1] # [None, 13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
                                        data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size] = defaultdict(list)
                                        for shuffle_idx in range(10):
                                            for i in range(n_models):
                                                # print(f"{i}, {shuffle_idx}")
                                                for test_sample_batch_idx in [0]:
                                                    # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/metrics_init.out"
                                                                                                                                                                        #  cwd_ML_with_data_4     _      1          _      0   _train_      0             shuffleidx_      0                       testsamplebatchidx_      4           nsamples_      32       njobs_      100            trees_      1       depth_      None         -      1                    rawinput_sampling1_      exact         treemethod_      1         maxbin_modize_par_      True            100  removesharedornoisestags_verpak
                                                    filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/False100_clustering/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_sim{sim_thr}assignment/metrics_init.out"
                                                                                                                                                                      #  cwd_ML_with_  data_4   _      50         _      49  _train_      0             shuffleidx_      0                       testsamplebatchidx_      4           nsamples_      32       njobs_      100            trees_      1       depth_      None         -      1                    rawinput_sampling1_      exact         treemethod_      1         maxbin_modize_par_      False           100  removesharedornoisestags_sim0.85assignment                                                    
                                                    if os.path.isfile(filename):
                                                        with open(filename) as file:
                                                            for line in file:
                                                                line_l = line.rstrip().split(":")
                                                                if line_l[0] == "BOW_XGB_init.fit":
                                                                    print(float(line_l[1]))
                                                                    data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["traintime"].append(float(line_l[1]))
                                                                if line_l[0] == "tagsets_to_matrix-trainset_xsize":
                                                                    data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["samplesize"].append(float(line_l[1]))
                                                                if line_l[0] == "tagsets_to_matrix-trainset_ysize":
                                                                    data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["dimensions"].append(float(line_l[1]))
                                                    else:
                                                        print(filename)

                                                    # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/index_label_mapping"
                                                    filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/False100_clustering/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_sim{sim_thr}assignment/index_label_mapping"
                                                                                                                                                                      #  cwd_ML_with_  data_4   _      50         _      49  _train_      0             shuffleidx_      0                       testsamplebatchidx_      4           nsamples_      32       njobs_      100            trees_      1       depth_      None         -      1                    rawinput_sampling1_      exact         treemethod_      1         maxbin_modize_par_      False           100  removesharedornoisestags_sim0.85assignment                                                    
                                                    if os.path.isfile(filename):
                                                        with open(filename, "rb") as file:
                                                            d = pickle.load(file)
                                                            data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["label_count"].append(len(d))
                                                            # print(f"submode list of labels: {d}")
                                                    else:
                                                        print(filename)

                                                    # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/intra_model_similarity.out"
                                                    # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_noverpak_cosinemean/similarity.out"
                                                    filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/False100_clustering/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_sim{sim_thr}assignment/intra_model_similarity.out"
                                                                                                                                                                        #  cwd_ML_with_  data_4   _      50         _      49  _train_      0             shuffleidx_      0                       testsamplebatchidx_      4           nsamples_      32       njobs_      100            trees_      1       depth_      None         -      1                    rawinput_sampling1_      exact         treemethod_      1         maxbin_modize_par_      False           100  removesharedornoisestags_sim0.85assignment                                                    
                                                    if os.path.isfile(filename):
                                                        with open(filename) as file:
                                                            d = json.load(file)
                                                            data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["cosine_sim"].append(float(d["Mean Cosine Similarity"]))
                                                    else:
                                                        print(filename)

                                            data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["traintime"].append(float("inf"))
                                            data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["samplesize"].append(float("inf"))
                                            data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["dimensions"].append(float("inf"))
                                            data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["cosine_sim"].append(float("inf"))
                                            data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["label_count"].append(float("inf"))
                                        # for shuffle_idx in range(3):
                                            for test_sample_batch_idx in [0]:
                                                # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak_on_demand_expert/metrics_pred.out"
                                                # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/metrics_pred.out"
                                                filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/False100_clustering/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_sim{sim_thr}assignment/metrics_pred.out"
                                                          # /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/            cwd_ML_with_data_4     _50               _train_  0                 shuffleidx_  0                           testsamplebatchidx_  4               nsamples_32             njobs_  32              clfnjobs_  100                trees_  1           depth_  None             -  1                        rawinput_sampling1_  exact             treemethod_  1             maxbin_modize_par_  False              100   removesharedornoisestags_sim0.85assignment

                                                if os.path.isfile(filename):
                                                    try:
                                                        with open(filename, "r") as f:
                                                            inferencetime_in_a_bag = 0
                                                            for line in f:
                                                                if line[:len(" BOW_XGB.predict_")] == " BOW_XGB.predict_":
                                                                    line_l = line.split(":")
                                                                    inferencetime_in_a_bag += round(float(line_l[1]),2)
                                                                elif line[:len(" BOW_XGB.predictclf")] == " BOW_XGB.predictclf":
                                                                    line_l = line.split(":")
                                                                    data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["inferencetime_clf"].append(round(float(line_l[1]),2))
                                                                else:
                                                                    line_l = line.rstrip().split()
                                                                    if len(line_l) == 9:
                                                                        data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["f1-score"].append((float(line_l[3])))
                                                                        # data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["f1-score-config"].append(str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx")
                                                                    if len(line_l) == 7:
                                                                        data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["precision"].append((float(line_l[1])))
                                                                    if len(line_l) == 8:
                                                                        data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["recall"].append((float(line_l[2])))
                                                            data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["inferencetime"].append(round(inferencetime_in_a_bag,2))
                                                    except Exception as e:
                                                        print(e)
                                                else:
                                                    print(filename)
                                                # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/inter_model_share_token.out"
                                                # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_noverpak_cosinemean/similarity.out"
                                                filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_sim{sim_thr}assignment/inter_model_share_token.out"
                                                            # /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/            cwd_ML_with_data_4     _50               _train_  0                 shuffleidx_  0                           testsamplebatchidx_  4               nsamples_32             njobs_  32              clfnjobs_  100                trees_  1           depth_  None             -  1                        rawinput_sampling1_  exact             treemethod_  1             maxbin_modize_par_  False              100   removesharedornoisestags_sim0.85assignment
                                                if os.path.isfile(filename):
                                                    with open(filename) as file:
                                                        d = json.load(file)
                                                        data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["model_share_token"].append(float(d["model_share_token"]))
                                                        data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["model_share_token_percent"].append(float(d["model_share_token_percent"]))
                                                else:
                                                    print(filename)

                                                    
for nmodel,datadict in data_d.items():
    print(nmodel, "!!!!!!!!!!!!!!!")
    for metric, value in datadict.items():
        print(metric, value)
        print()
    print()



# [36.364, 13.485, 11.404, 11.065, 26.566, 9.699, 10.865, 12.597, 8.459, 16.239, 21.427, 19.303, 93.203, 17.617, 8.294, 47.261, 12.097, 11.627, 41.329, 15.604, 15.766, 18.41, 10.327, 66.757, 16.69, 17.69, 20.05, 46.571, 13.465, 19.513, 8.022, 17.94, 18.002, 13.92, 22.415, 15.509, 12.552, 18.035, 9.757, 27.081, 73.474, 47.593, 74.111, 21.285, 15.856, 16.16, 17.217, 69.184, 14.446, 8.693, 45.536, 10.11, 8.035, 46.619, 46.403, 18.114, 14.661, 28.029, 25.802, 20.253, 10.724, 9.285, 11.087, 44.712, 9.392, 54.832, 44.561, 17.115, 13.226, 12.435, 87.071, 30.016, 16.584, 16.399, 17.257, 55.64, 40.306, 53.774, 37.759, 11.961, 9.593, 36.385, 5.516, 16.508, 78.805, 17.085, 50.567, 41.234, 13.825, 13.542, 12.482, 18.305, 21.051, 32.199, 6.929, 7.575, 6.889, 86.014, 42.648, 18.265, 13.387, 24.348, 51.386, 55.725, 42.145, 83.474, 66.209, 15.028, 66.29, 53.377, 22.374, 32.669, 21.477, 54.188, 30.603, 8.358, 8.899, 63.631, 27.14, 24.506, 29.812, 21.633, 60.979, 17.555, 24.368, 49.808, 35.467, 5.452, 18.488, 33.757, 46.649, 17.287, 45.546, 11.432, 26.753, 23.309, 76.478, 23.495, 14.661, 17.239, 12.951, 10.492, 36.383, 9.269, 50.861, 13.58, 20.092, 11.257, 16.103, 18.762, 31.825, 28.164, 15.551, 29.52, 14.614, 22.147, 9.767, 54.061, 8.305, 36.086, 37.786, 16.561, 18.082, 17.284, 54.846, 19.994, 10.264, 7.539, 53.3, 22.231, 10.023, 5.978, 24.261, 10.221, 25.249, 23.334, 13.786, 13.331, 21.289, 11.961, 12.055, 24.873, 13.518, 34.735, 19.93, 7.59, 37.403, 12.493, 7.529, 13.97, 14.283, 25.268, 38.448, 136.859, 16.567, 9.997, 26.482, 33.504, 12.666, 19.982, 15.713, 24.247, 5.948, 23.838, 16.819, 106.666, 28.972, 18.387, 12.011, 7.986, 18.694, 20.941, 29.089, 21.707, 35.23, 7.125, 25.367, 14.327, 17.039, 22.512, 23.98, 19.555, 18.463, 8.103, 36.176, 15.404, 20.486, 15.609, 10.324, 9.591, 7.659, 20.494, 9.631, 20.562, 21.879, 41.913, 26.405, 10.752, 8.154, 62.254, 9.228, 8.6, 33.313, 52.982, 9.24, 14.567, 23.194, 12.711, 33.539, 22.07]
















# train time
# train_times_l = []
# sum_train_times_l = []
# for n_models in [50,25,20,15,10,5,1]: 
#     train_time_l = []
#     sum_train_time = 0
#     for model_idx in range(n_models):
#                 #    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_             50  _45                _train_64njobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par_removesharedornoisestags"
#         # filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_"+str(n_models)+"_"+str(model_idx)+"_train_8njobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par_removesharedornoisestags0/metrics_init.out"
#         filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_"+str(n_models)+"_"+str(model_idx)+"_train_64njobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par/metrics_init_0.out"
#         if os.path.isfile(filename):
            
#             with open(filename) as file:
#                 for line in file:
#                     line_l = line.rstrip().split(":")
#                     if line_l[0] == "BOW_XGB_init.fit":
#                         train_time_l.append(float(line_l[1]))
#                         sum_train_time += float(line_l[1])
#     train_times_l.append(train_time_l)
#     sum_train_times_l.append(round(sum_train_time, 2))
# print(train_times_l)
# print()
# print(sum_train_times_l)
# # print([train_time_l[0] for train_time_l in train_times_l])


# # # Print sum training time of each N models
# for input_size in [None]:#[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]: # [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
#     inference_times_l = []
#     for n_models in [50,25,20,15,10,5,1]:
#         inference_time = 0
#         for model_idx in range(n_models):
#             # filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_"+str(n_models)+"_"+str(model_idx)+"_train_64njobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par_removesharedornoisestags/metrics_init.out"
#             filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_"+str(n_models)+"_"+str(model_idx)+"_train_64njobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par/metrics_init_0.out"
#             if os.path.isfile(filename):
#                 with open(filename) as file:
#                     for line in file:
#                         line_l = line.rstrip().split(":")
#                         if line_l[0] == "BOW_XGB_init.fit":
#                             inference_time+=float(line_l[1])
#         else:
#             inference_times_l.append(round(inference_time,2))
#     print(inference_times_l)


# # Print training time of each model
# for input_size in [None]:#[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]: # [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
#     inference_times_l = []
#     for n_models in [50,25,20,15,10,5,1]:
#         inference_time_l = []
#         for model_idx in range(n_models):
#             # filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_"+str(n_models)+"_"+str(model_idx)+"_train_8njobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par_removesharedornoisestags0/metrics_init.out"
#             filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_"+str(n_models)+"_"+str(model_idx)+"_train_64njobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par/metrics_init_0.out"
#             time_counter = 0
#             if os.path.isfile(filename):
#                 with open(filename) as file:
#                     for line in file:
#                         line_l = line.rstrip().split(":")
#                         if line_l[0] == "BOW_XGB_init.fit":
#                             inference_time_l.append(round(float(line_l[1]),2))
#                             time_counter+=1
#         else:
#             inference_time_l.extend([0]*(46-time_counter))
#             # print("+".join(inference_time_l))
#             inference_times_l.append(inference_time_l)
#     inference_times_l = list(map(list, zip(*inference_times_l)))
#     # print(inference_times_l)
#     for inference_time_l in inference_times_l:
#         print(inference_time_l, ",")
#     # print("["+", ".join(inference_times_l)+"],")




# # inference time
# inference_times_l = []
# for input_size in [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]: # [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
#     filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_4_train_8njobs_100trees_1depth_"+str(input_size)+"rawinput_sampling1_exacttreemethod/metrics_pred.out"
#     inference_time_l = []
#     with open(filename) as file:
#         for line in file:
#             line_l = line.rstrip().split(":")
#             if line_l[0] == " BOW_XGB.predict":
#                 inference_time_l.append(line_l[1])
#     # print("+".join(inference_time_l))
#     inference_times_l.append("+".join(inference_time_l))
# print(", ".join(inference_times_l))


# # inference time based on input dimensions
# for n_models in [8,4,2,1]:
#     inference_times_l = []
#     for input_size in [13664, 27329, 54659, 109319]:#[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]: # [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
#                     # /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_                8_train_8njobs_100trees_1depth_               Nonerawinput_sampling1_exacttreemethod_1maxbin
#         filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_"+str(n_models)+"_train_8njobs_100trees_1depth_"+str(input_size)+"rawinput_sampling1_exacttreemethod/metrics_pred.out"
#         inference_time_l = []
#         with open(filename) as file:
#             for line in file:
#                 line_l = line.rstrip().split(":")
#                 if line_l[0] == " BOW_XGB.predict":
#                     inference_time_l.append(line_l[1])
#         # print("+".join(inference_time_l))
#         inference_times_l.append("+".join(inference_time_l))
#     print("["+", ".join(inference_times_l)+"],")


# # Read Inference Time: based on number of labels per model
# for input_size in [None]:#[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]: # [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
#     inference_times_l = []
#     for n_models in [8,4,2,1]:
#                     # /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_                8_train_8njobs_100trees_1depth_               Nonerawinput_sampling1_exacttreemethod_1maxbin
#                     # /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_                8_train_1njobs_8clfnjobs_100trees_1depth_               Nonerawinput_sampling1_exacttreemethod_1maxbin
#         filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_"+str(n_models)+"_train_1njobs_8clfnjobs_100trees_1depth_"+str(input_size)+"rawinput_sampling1_exacttreemethod_1maxbin/metrics_pred.out"
#         inference_time_l = []
#         time_counter = 0
#         with open(filename) as file:
#             for line in file:
#                 line_l = line.rstrip().split(":")
#                 if line_l[0] == " BOW_XGB.predict":
#                     inference_time_l.append(float(line_l[1]))
#                     time_counter+=1
#             else:
#                 inference_time_l.extend([0]*(8-time_counter))
#         # print("+".join(inference_time_l))
#         inference_times_l.append(inference_time_l)
#     inference_times_l = list(map(list, zip(*inference_times_l)))
#     # print(inference_times_l)
#     for inference_time_l in inference_times_l:
#         print(inference_time_l)
#     # print("["+", ".join(inference_times_l)+"],")



# # Read Input Dimensions: based on number of labels per model
# for input_size in [None]:
#     inference_times_l = []
#     for n_models in [5]:
#         inference_time_l = []
#         for i in range(n_models):
#                         # /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_              5  _      0   _train_8njobs_100trees_1depth_            None   /1rawinput_sampling1_exacttreemethod_1maxbin_modize_par/metrics_init_0.out
#             filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_"+str(n_models)+"_"+str(i)+"_train_8njobs_100trees_1depth_"+str(input_size)+"/1rawinput_sampling1_exacttreemethod_1maxbin_modize_par/metrics_init_0.out"
#             with open(filename) as file:
#                 for line in file:
#                     line_l = line.rstrip().split(":")
#                     if line_l[0] == "tagsets_to_matrix-trainset_ysize":
#                         inference_time_l.append(float(line_l[1]))
#         inference_times_l.append(inference_time_l)
#     # inference_times_l.append([])
#     inference_times_l = list(map(lambda x: sum(x), inference_times_l))
#     for inference_time_l in inference_times_l:
#         print(inference_time_l)


# # training time of 500 labels vs 80 labels: 12500*199726.0*500/(2000*109319*80)*1200/60/60*8/64 ~= 3hr

# # read train F1-Scores
# for input_size in [None]:#[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]: # [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
#     # by_Nmodels_l = []
#     for n_models in [50]:
#         by_eachmodel_l = []
#         for i in range(n_models):
#                                                                                                                     #cwd_ML_with_data_3_            50   _      15  _train_64njobs_100trees_1depth_None-               1rawinput_sampling1_exacttreemethod_1maxbin_modize_par_removesharedtags
#             # filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_"+str(n_models)+"_"+str(i)+"_train_64njobs_100trees_1depth_"+str(input_size)+"-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par_removesharedtags/metrics_init.out"
#                                                                                                                     #cwd_ML_with_data_3_            50   _      49  _train_64njobs_100trees_1depth_            None   -1rawinput_sampling1_exacttreemethod_1maxbin_modize_par
#             filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_"+str(n_models)+"_"+str(i)+"_train_64njobs_100trees_1depth_"+str(input_size)+"-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par/metrics_init_0.out"
#             if os.path.isfile(filename):
#                 with open(filename) as file:
#                     for line in file:
#                         line_l = line.rstrip().split(":")
#                         if line_l[0] == "F1 SCORE ":
#                             by_eachmodel_l.append(line_l[1])
#         for idx, by_eachmodel in enumerate(by_eachmodel_l):
#             print(idx, by_eachmodel)

# # read test precisions for each label
# interested_lines_l, interested_precisions_l, interested_pkgs_l = [], [], []
# # filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_50_train_1njobs_64clfnjobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par/metrics_pred.out"
# filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/data_0_results/cwd_ML_with_data_0_8_train_8njobs_100trees_1depth_Nonerawinput_sampling1_exacttreemethod_1maxbin/metrics_pred.out"
# with open(filename) as file:
#     for line in file:
#         line_l = line.split()
#         if len(line_l) == 5 and float(line_l[1])<0.2:
#             interested_lines_l.append(line.rstrip())
#             interested_precisions_l.append(float(line_l[1]))
#             interested_pkgs_l.append(line_l[0])
# print(len(interested_lines_l))
# for interested_pkg, interested_line, interested_precisions in zip(interested_pkgs_l, interested_lines_l,interested_precisions_l):
#     print(interested_pkg, interested_line, interested_precisions)
# print(interested_pkgs_l)

# # read weighted test f1-score
# interested_lines_l, interested_value_l = [], []
# # for n_models in [50,25,20,15,10,5,1]:
# for n_models in [8,4,2,1]:
#     # filename =   "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_"+str(n_models)+"_train_64njobs_64clfnjobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par_removesharedornoisestags/metrics_pred.out"
#     # filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_3_"+str(n_models)+"_train_1njobs_64clfnjobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par/metrics_pred.out"
#     filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_"+str(n_models)+"_train_8njobs_8clfnjobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par_removesharedornoisestags0/metrics_pred.out"
#     # filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/data_0_results/cwd_ML_with_data_0_"+str(n_models)+"_train_8njobs_100trees_1depth_Nonerawinput_sampling1_exacttreemethod_1maxbin/metrics_pred.out"
#     with open(filename) as file:
#         for line in file:
#             line_l = line.split()
#             if len(line_l) == 7:
#                 interested_lines_l.append(line.rstrip())
#                 interested_value_l.append(float(line_l[1]))
# print(interested_value_l)
# # print(len(interested_lines_l))
# # for interested_line, interested_value in zip(interested_lines_l,interested_value_l):
# #     print(interested_line, interested_value)