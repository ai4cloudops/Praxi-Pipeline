import os
from collections import defaultdict


# Read all train data
data_d = dict()
n_samples = 25
for dataset in ["data_3"]:
    for n_jobs in [32]:
        for clf_njobs in [32]:
            for n_models, test_batch_count in zip([25, 10, 1],[1, 1, 8]): #([50,25,20,15,10,5,1],[1,1,1,1,1,1,8]): # ([1,25,10],[8,1,1])
                data_d[str(n_models)+"_models"] = defaultdict(list)
                for n_estimators in [100]:
                    for depth in [1]:
                        for tree_method in["exact"]: # "exact","approx","hist"
                            for max_bin in [1]:
                                for input_size, dim_compact_factor in zip([10000, 1000, 500, 5000, 15000],[1,1,1,1,1,1]): # [None, 10000, 1000, 500, 5000, 15000],[1,1,1,1,1,1] # [None, 13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
                                    for i in range(n_models):
                                        
                                        for shuffle_idx in range(3):
                                            for test_sample_batch_idx in [4]:
                                                filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"_shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_removesharedornoisestags/metrics_init.out"
                                                if os.path.isfile(filename):
                                                    with open(filename) as file:
                                                        for line in file:
                                                            line_l = line.rstrip().split(":")
                                                            if line_l[0] == "BOW_XGB_init.fit":
                                                                data_d[str(n_models)+"_models"]["traintime"].append(float(line_l[1]))
                                                            if line_l[0] == "tagsets_to_matrix-trainset_xsize":
                                                                data_d[str(n_models)+"_models"]["samplesize"].append(float(line_l[1]))
                                                            if line_l[0] == "tagsets_to_matrix-trainset_ysize":
                                                                data_d[str(n_models)+"_models"]["dimensions"].append(float(line_l[1]))
                                                # cwd =      "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"_shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_removesharedornoisestags/"
                                                filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"_shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_removesharedornoisestags/metrics_pred.out"
                                                if os.path.isfile(filename):
                                                    try:
                                                        with open(filename, "r") as f:
                                                            for line in f:
                                                                line_l = line.rstrip().split()
                                                                if len(line_l) == 9:
                                                                    data_d[str(n_models)+"_models"]["f1-score"].append((float(line_l[3])))
                                                    except Exception as e:
                                                        print(e)

                                                    
for nmodel,datadict in data_d.items():
    print(nmodel)
    for metric, value in datadict.items():
        print(metric, value)
        print()
    print()


















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